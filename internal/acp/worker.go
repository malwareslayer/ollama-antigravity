package acp

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"ollama-antigravity/internal/config"
)

// ErrNativeToolBlocked indicates agy attempted an internal native tool call in headless mode.
var ErrNativeToolBlocked = errors.New("native antigravity tool invocation is disabled")

// PromptRequest defines a single inference job for an ACP worker.
type PromptRequest struct {
	ModelID      string
	SystemPrompt string
	Prompt       []ContentBlock
}

// StreamEventType classifies events emitted during prompt execution.
type StreamEventType int

const (
	EventTextChunk StreamEventType = iota
	EventThoughtChunk
	EventDone
	EventError
)

// StreamEvent carries incremental output or terminal status from a worker.
type StreamEvent struct {
	Type    StreamEventType
	Text    string
	Usage   Usage
	Err     error
	LoadDur time.Duration
	EvalDur time.Duration
}

type workerJob struct {
	ctx    context.Context
	req    PromptRequest
	events chan StreamEvent
}

type rpcEvent struct {
	msg RPCMessage
	err error
}

type childProc struct {
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	rpcCh      chan rpcEvent
	readerDone chan struct{}
	stderrDone chan struct{}
}

type worker struct {
	id     int
	cfg    config.Config
	logger *slog.Logger
	pool   *Pool

	jobs   chan *workerJob
	stopCh chan struct{}
	doneCh chan struct{}

	proc   *childProc
	nextID int64

	sessionID      string
	sessionModel   string
	sessionSysHash string
	sessionTurns   int
	sessionTokens  int
}

func newWorker(id int, cfg config.Config, logger *slog.Logger, pool *Pool) *worker {
	return &worker{
		id:     id,
		cfg:    cfg,
		logger: logger.With("worker", id),
		pool:   pool,
		jobs:   make(chan *workerJob),
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}
}

func (w *worker) start(initDone chan<- error) {
	go w.loop(initDone)
}

func (w *worker) stop() {
	select {
	case <-w.stopCh:
	default:
		close(w.stopCh)
	}
	<-w.doneCh
}

// loop runs for the entire lifetime of the worker on a dedicated, OS-locked thread
// so Linux PR_SET_PDEATHSIG reliably terminates the child if the bridge dies.
func (w *worker) loop(initDone chan<- error) {
	runtime.LockOSThread()
	defer close(w.doneCh)
	defer w.killAndReap()

	models, err := w.spawnAndBootstrap()
	if err != nil {
		w.logger.Error("initial worker spawn failed", "err", err)
		initDone <- err
	} else {
		w.pool.updateCatalog(models)
		initDone <- nil
	}

	w.pool.release(w)

	for {
		select {
		case <-w.stopCh:
			return
		case job := <-w.jobs:
			w.runJob(job)
			select {
			case <-w.stopCh:
				return
			default:
				w.pool.release(w)
			}
		}
	}
}

func (w *worker) runJob(job *workerJob) {
	defer close(job.events)

	start := time.Now()
	if w.proc == nil {
		if err := w.spawnAndInit(); err != nil {
			w.emit(job.ctx, job.events, StreamEvent{
				Type: EventError,
				Err:  fmt.Errorf("acp spawn failed: %w", err),
			})
			return
		}
	}

	// Hard deadline covers the full request (session/new + set_model + prompt + close).
	timeoutCtx, cancel := context.WithTimeout(job.ctx, w.cfg.Timeout)
	defer cancel()

	usage, evalDur, dirty, err := w.executeSession(timeoutCtx, job.req, job.events)
	loadDur := time.Since(start) - evalDur
	if loadDur < 0 {
		loadDur = 0
	}

	if err != nil {
		w.logger.Warn("acp job failed, respawning worker", "err", err, "dirty", dirty)
		w.killAndReap()
		if respawnErr := w.spawnAndInit(); respawnErr != nil {
			w.logger.Error("worker respawn failed after job error", "err", respawnErr)
		}
		w.emit(context.Background(), job.events, StreamEvent{
			Type: EventError,
			Err:  err,
		})
		return
	}

	if dirty {
		w.killAndReap()
		if respawnErr := w.spawnAndInit(); respawnErr != nil {
			w.logger.Error("worker respawn failed after dirty session", "err", respawnErr)
		}
	}

	w.emit(job.ctx, job.events, StreamEvent{
		Type:    EventDone,
		Usage:   usage,
		LoadDur: loadDur,
		EvalDur: evalDur,
	})
}

func (w *worker) emit(ctx context.Context, ch chan<- StreamEvent, ev StreamEvent) {
	select {
	case ch <- ev:
	case <-ctx.Done():
	case <-w.stopCh:
	}
}

func (w *worker) spawnAndBootstrap() ([]AvailableModel, error) {
	if err := w.spawnAndInit(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := w.callSessionNew(ctx, "")
	if err != nil {
		w.killAndReap()
		return nil, err
	}
	if err := w.callSessionClose(ctx, res.SessionID); err != nil {
		w.logger.Debug("bootstrap session/close failed", "err", err)
	}
	return res.Models.AvailableModels, nil
}

func (w *worker) spawnAndInit() error {
	w.killAndReap()

	args := []string{"acp"}
	if w.cfg.DisableTelemetry {
		args = append(args, "--disable-telemetry")
	}
	if !w.cfg.IncludeDefaultCustom {
		args = append(args, "--include-default-customizations=false")
	}
	if w.cfg.Sandbox {
		args = append(args, "--sandbox")
	}

	cmd := exec.Command(w.cfg.AgyBin, args...)
	cmd.Dir = w.cfg.EmptyCWD
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Pdeathsig: syscall.SIGKILL,
		Setpgid:   true,
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return fmt.Errorf("start %s: %w", w.cfg.AgyBin, err)
	}

	proc := &childProc{
		cmd:        cmd,
		stdin:      stdin,
		rpcCh:      make(chan rpcEvent, 64),
		readerDone: make(chan struct{}),
		stderrDone: make(chan struct{}),
	}

	go w.readStdout(stdout, proc.rpcCh, proc.readerDone)
	go w.drainStderr(stderr, proc.stderrDone)

	w.proc = proc
	w.nextID = 0

	initCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := w.callInitialize(initCtx); err != nil {
		w.killAndReap()
		return fmt.Errorf("initialize handshake: %w", err)
	}

	w.logger.Info("agy acp worker initialized", "pid", cmd.Process.Pid)
	return nil
}

func (w *worker) readStdout(r io.Reader, out chan<- rpcEvent, done chan<- struct{}) {
	defer close(done)
	defer close(out)

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var msg RPCMessage
		if err := json.Unmarshal([]byte(line), &msg); err != nil {
			out <- rpcEvent{err: fmt.Errorf("invalid json-rpc line: %w", err)}
			return
		}
		out <- rpcEvent{msg: msg}
	}

	if err := scanner.Err(); err != nil {
		out <- rpcEvent{err: fmt.Errorf("stdout scanner: %w", err)}
		return
	}
	out <- rpcEvent{err: io.EOF}
}

func (w *worker) drainStderr(r io.Reader, done chan<- struct{}) {
	defer close(done)

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 16*1024), 1024*1024)
	for scanner.Scan() {
		w.logger.Debug("agy stderr", "line", scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		w.logger.Debug("agy stderr read ended", "err", err)
	}
}

func (w *worker) resetSessionState() {
	w.sessionID = ""
	w.sessionModel = ""
	w.sessionSysHash = ""
	w.sessionTurns = 0
	w.sessionTokens = 0
}

func (w *worker) killAndReap() {
	w.resetSessionState()
	proc := w.proc
	if proc == nil {
		return
	}
	w.proc = nil

	if proc.stdin != nil {
		_ = proc.stdin.Close()
	}
	if proc.cmd != nil && proc.cmd.Process != nil {
		pid := proc.cmd.Process.Pid
		if pid > 0 {
			_ = syscall.Kill(-pid, syscall.SIGKILL)
		}
		_ = proc.cmd.Process.Kill()
	}

	<-proc.readerDone
	<-proc.stderrDone

	if proc.cmd != nil {
		if err := proc.cmd.Wait(); err != nil {
			w.logger.Debug("reaped agy acp process", "err", err)
		}
	}
}

func (w *worker) executeSession(
	ctx context.Context,
	req PromptRequest,
	events chan<- StreamEvent,
) (usage Usage, evalDur time.Duration, dirty bool, err error) {
	sysHash := hashSystemPrompt(req.SystemPrompt)
	canReuse := w.cfg.SessionReuse &&
		w.sessionID != "" &&
		w.sessionSysHash == sysHash &&
		w.sessionTurns < w.cfg.SessionMaxTurns &&
		w.sessionTokens < w.cfg.SessionMaxTokens

	if !canReuse {
		if w.sessionID != "" {
			closeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_ = w.callSessionClose(closeCtx, w.sessionID)
			cancel()
			w.resetSessionState()
		}

		newRes, err := w.callSessionNew(ctx, req.SystemPrompt)
		if err != nil {
			return Usage{}, 0, true, fmt.Errorf("session/new: %w", err)
		}
		w.pool.updateCatalog(newRes.Models.AvailableModels)
		w.sessionID = newRes.SessionID
		w.sessionModel = newRes.Models.CurrentModelID
		w.sessionSysHash = sysHash
		w.sessionTurns = 0
		w.sessionTokens = 0
	}

	if req.ModelID != "" && req.ModelID != w.sessionModel {
		if err := w.callSessionSetModel(ctx, w.sessionID, req.ModelID); err != nil {
			w.resetSessionState()
			return Usage{}, 0, true, fmt.Errorf("session/set_model(%s): %w", req.ModelID, err)
		}
		w.sessionModel = req.ModelID
	}

	evalStart := time.Now()
	promptRes, err := w.callSessionPrompt(ctx, w.sessionID, req.Prompt, events)
	evalDur = time.Since(evalStart)
	if err != nil {
		w.resetSessionState()
		return Usage{}, evalDur, true, fmt.Errorf("session/prompt: %w", err)
	}

	w.sessionTurns++
	w.sessionTokens = promptRes.Usage.TotalTokens

	if !w.cfg.SessionReuse || w.sessionTurns >= w.cfg.SessionMaxTurns || w.sessionTokens >= w.cfg.SessionMaxTokens {
		closeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		closeErr := w.callSessionClose(closeCtx, w.sessionID)
		cancel()
		w.resetSessionState()
		if closeErr != nil {
			w.logger.Warn("session/close failed, marking worker dirty", "err", closeErr)
			return promptRes.Usage, evalDur, true, nil
		}
	}

	return promptRes.Usage, evalDur, false, nil
}

func hashSystemPrompt(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func (w *worker) callInitialize(ctx context.Context) error {
	params := initializeParams{
		ProtocolVersion:    1,
		ClientCapabilities: map[string]any{},
		ClientInfo: clientInfo{
			Name:    "acp-ollama-bridge",
			Version: "1.0.0",
		},
	}
	_, err := w.rpcCall(ctx, "initialize", params, nil)
	return err
}

func (w *worker) callSessionNew(ctx context.Context, systemAppend string) (sessionNewResult, error) {
	params := sessionNewParams{
		CWD:        w.cfg.EmptyCWD,
		MCPServers: []any{},
	}
	if strings.TrimSpace(systemAppend) != "" {
		params.Meta = &sessionNewMeta{
			Antigravity: &antigravityMeta{
				SystemPrompt: &systemPromptMeta{
					Append: systemAppend,
				},
			},
		}
	}
	raw, err := w.rpcCall(ctx, "session/new", params, nil)
	if err != nil {
		return sessionNewResult{}, err
	}
	var res sessionNewResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return sessionNewResult{}, fmt.Errorf("decode session/new result: %w", err)
	}
	if res.SessionID == "" {
		return sessionNewResult{}, errors.New("session/new returned empty sessionId")
	}
	return res, nil
}

func (w *worker) callSessionSetModel(ctx context.Context, sid, modelID string) error {
	_, err := w.rpcCall(ctx, "session/set_model", sessionSetModelParams{
		SessionID: sid,
		ModelID:   modelID,
	}, nil)
	return err
}

func (w *worker) callSessionPrompt(
	ctx context.Context,
	sid string,
	prompt []ContentBlock,
	events chan<- StreamEvent,
) (sessionPromptResult, error) {
	raw, err := w.rpcCall(ctx, "session/prompt", sessionPromptParams{
		SessionID: sid,
		Prompt:    prompt,
	}, func(msg RPCMessage) error {
		if msg.Method != "session/update" {
			return nil
		}
		var upd sessionUpdateParams
		if err := json.Unmarshal(msg.Params, &upd); err != nil {
			w.logger.Debug("failed to decode session/update", "err", err)
			return nil
		}
		switch upd.Update.SessionUpdate {
		case "tool_call", "tool_call_update":
			if w.cfg.RestrictNativeTools {
				toolName := strings.TrimSpace(upd.Update.Name)
				if toolName == "" {
					toolName = "unknown"
				}
				w.logger.Warn("antigravity attempted native tool invocation; neutralizing",
					"tool", toolName,
					"tool_call_id", upd.Update.ToolCallID,
					"session_update", upd.Update.SessionUpdate,
				)
				return fmt.Errorf("%w: %s", ErrNativeToolBlocked, toolName)
			}
		case "agent_message_chunk":
			if upd.Update.Content.Text != "" {
				w.emit(ctx, events, StreamEvent{
					Type: EventTextChunk,
					Text: upd.Update.Content.Text,
				})
			}
		case "agent_thought_chunk":
			if upd.Update.Content.Text != "" {
				w.emit(ctx, events, StreamEvent{
					Type: EventThoughtChunk,
					Text: upd.Update.Content.Text,
				})
			}
		}
		return nil
	})
	if err != nil {
		return sessionPromptResult{}, err
	}
	var res sessionPromptResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return sessionPromptResult{}, fmt.Errorf("decode session/prompt result: %w", err)
	}
	return res, nil
}

func (w *worker) callSessionClose(ctx context.Context, sid string) error {
	_, err := w.rpcCall(ctx, "session/close", sessionCloseParams{SessionID: sid}, nil)
	return err
}

func (w *worker) rpcCall(
	ctx context.Context,
	method string,
	params any,
	onNotif func(RPCMessage) error,
) (json.RawMessage, error) {
	if w.proc == nil {
		return nil, errors.New("no active acp subprocess")
	}

	w.nextID++
	reqID := w.nextID

	payload, err := json.Marshal(rpcRequest{
		JSONRPC: "2.0",
		ID:      reqID,
		Method:  method,
		Params:  params,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal %s: %w", method, err)
	}
	payload = append(payload, '\n')

	if _, err := w.proc.stdin.Write(payload); err != nil {
		return nil, fmt.Errorf("write %s: %w", method, err)
	}

	for {
		select {
		case <-w.stopCh:
			return nil, errors.New("worker shutting down")
		case <-ctx.Done():
			return nil, ctx.Err()
		case ev, ok := <-w.proc.rpcCh:
			if !ok {
				return nil, io.EOF
			}
			if ev.err != nil {
				return nil, ev.err
			}
			msg := ev.msg

			if msg.Method != "" {
				if isNonNullID(msg.ID) {
					w.handleServerRequest(msg)
					continue
				}
				if onNotif != nil {
					if err := onNotif(msg); err != nil {
						return nil, err
					}
				}
				continue
			}

			id, ok := parseRPCID(msg.ID)
			if !ok || id != reqID {
				w.logger.Debug("ignoring unmatched json-rpc response", "gotID", string(msg.ID), "wantID", reqID)
				continue
			}
			if msg.Error != nil {
				return nil, msg.Error
			}
			return msg.Result, nil
		}
	}
}

func (w *worker) handleServerRequest(msg RPCMessage) {
	resp := rpcResponse{
		JSONRPC: "2.0",
		ID:      msg.ID,
	}
	if msg.Method == "session/request_permission" {
		var p permissionRequestParams
		_ = json.Unmarshal(msg.Params, &p)
		var rejectID string
		for _, opt := range p.Options {
			if strings.Contains(strings.ToLower(opt.Kind), "reject") {
				rejectID = opt.OptionID
				break
			}
		}
		if rejectID != "" {
			resp.Result = map[string]any{
				"outcome": map[string]any{
					"outcome":  "selected",
					"optionId": rejectID,
				},
			}
		} else {
			resp.Result = map[string]any{
				"outcome": map[string]any{
					"outcome": "cancelled",
				},
			}
		}
	} else {
		resp.Error = &RPCError{
			Code:    -32601,
			Message: fmt.Sprintf("unsupported client method %q", msg.Method),
		}
	}

	b, err := json.Marshal(resp)
	if err != nil {
		w.logger.Debug("failed to marshal server request response", "err", err)
		return
	}
	b = append(b, '\n')
	if w.proc != nil && w.proc.stdin != nil {
		_, _ = w.proc.stdin.Write(b)
	}
}

func isNonNullID(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s != "" && s != "null"
}

func parseRPCID(raw json.RawMessage) (int64, bool) {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return 0, false
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}
