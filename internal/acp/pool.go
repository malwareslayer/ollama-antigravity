package acp

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"ollama-antigravity/internal/config"
)

var (
	ErrPoolClosed    = errors.New("acp worker pool is closed")
	ErrModelExcluded = errors.New("requested model is excluded")
	ErrModelNotFound = errors.New("requested model not found")
)

// Executor defines the inference & model catalog contract used by the HTTP layer.
type Executor interface {
	ListModels() []AvailableModel
	ResolveModel(requested string) (AvailableModel, error)
	Execute(ctx context.Context, req PromptRequest) (<-chan StreamEvent, error)
}

// Pool manages a fixed-size pool of OS-thread-locked agy acp workers.
type Pool struct {
	cfg      config.Config
	logger   *slog.Logger
	idle     chan *worker
	workers  []*worker
	closedCh chan struct{}

	closeOnce sync.Once

	mu             sync.RWMutex
	catalog        []AvailableModel
	allRaw         []AvailableModel
	exhaustedUntil map[string]time.Time
}

// NewPool constructs a worker pool without starting subprocesses yet.
func NewPool(cfg config.Config, logger *slog.Logger) *Pool {
	return &Pool{
		cfg:            cfg,
		logger:         logger,
		idle:           make(chan *worker, cfg.Workers),
		workers:        make([]*worker, 0, cfg.Workers),
		closedCh:       make(chan struct{}),
		exhaustedUntil: make(map[string]time.Time),
	}
}

// Start ensures ACP_EMPTY_CWD exists, spawns all workers, and waits for initial handshakes.
func (p *Pool) Start(ctx context.Context) error {
	if err := os.MkdirAll(p.cfg.EmptyCWD, 0o755); err != nil {
		return fmt.Errorf("create ACP_EMPTY_CWD %q: %w", p.cfg.EmptyCWD, err)
	}

	initCh := make(chan error, p.cfg.Workers)
	for i := 0; i < p.cfg.Workers; i++ {
		w := newWorker(i, p.cfg, p.logger, p)
		p.workers = append(p.workers, w)
		w.start(initCh)
	}

	var successes int
	var firstErr error
	for i := 0; i < p.cfg.Workers; i++ {
		select {
		case err := <-initCh:
			if err == nil {
				successes++
			} else if firstErr == nil {
				firstErr = err
			}
		case <-ctx.Done():
			p.Stop()
			return ctx.Err()
		}
	}

	if successes == 0 {
		p.Stop()
		return fmt.Errorf("all %d acp workers failed to initialize: %w", p.cfg.Workers, firstErr)
	}

	p.logger.Info("acp worker pool ready",
		"workers_ready", successes,
		"workers_total", p.cfg.Workers,
		"models_available", len(p.ListModels()),
	)
	return nil
}

// Stop terminates all workers and reaps their child processes.
func (p *Pool) Stop() {
	p.closeOnce.Do(func() {
		close(p.closedCh)
		for _, w := range p.workers {
			w.stop()
		}
	})
}

func (p *Pool) release(w *worker) {
	select {
	case <-p.closedCh:
	case p.idle <- w:
	}
}

func (p *Pool) updateCatalog(raw []AvailableModel) {
	if len(raw) == 0 {
		return
	}
	filtered := make([]AvailableModel, 0, len(raw))
	rawCopy := make([]AvailableModel, 0, len(raw))
	for _, m := range raw {
		rawCopy = append(rawCopy, m)
		if !IsExcludedModel(m, p.cfg.ExcludeModels) {
			filtered = append(filtered, m)
		}
	}

	p.mu.Lock()
	p.allRaw = rawCopy
	p.catalog = filtered
	p.mu.Unlock()
}

// IsExcludedModel checks whether a model matches any configured exclusion rule.
func IsExcludedModel(m AvailableModel, excludes []string) bool {
	idLow := strings.ToLower(m.ModelID)
	nameLow := strings.ToLower(m.Name)
	for _, ex := range excludes {
		pat := strings.ToLower(strings.TrimSpace(ex))
		if pat == "" {
			continue
		}
		if idLow == pat || strings.Contains(idLow, pat) || strings.Contains(nameLow, pat) {
			return true
		}
	}
	return false
}

// ListModels returns a copy of all non-excluded models discovered from agy acp.
func (p *Pool) ListModels() []AvailableModel {
	p.mu.RLock()
	defer p.mu.RUnlock()

	out := make([]AvailableModel, len(p.catalog))
	copy(out, p.catalog)
	return out
}

// ResolveModel maps an Ollama model name (with optional :latest or short alias)
// to a concrete non-excluded ACP modelId.
func (p *Pool) ResolveModel(requested string) (AvailableModel, error) {
	req := strings.TrimSpace(requested)
	if req == "" {
		req = p.cfg.DefaultModel
	}

	p.mu.RLock()
	catalog := make([]AvailableModel, len(p.catalog))
	copy(catalog, p.catalog)
	allRaw := make([]AvailableModel, len(p.allRaw))
	copy(allRaw, p.allRaw)
	p.mu.RUnlock()

	return ResolveFromCatalog(req, catalog, allRaw, p.cfg.ExcludeModels)
}

// ResolveFromCatalog performs deterministic model resolution and exclusion checks.
func ResolveFromCatalog(
	req string,
	catalog []AvailableModel,
	allRaw []AvailableModel,
	excludes []string,
) (AvailableModel, error) {
	trimmed := strings.TrimSpace(req)
	noLatest := strings.TrimSuffix(trimmed, ":latest")
	low := strings.ToLower(noLatest)

	// Explicit exclusion check first so excluded models are never reached via fallback.
	for _, m := range allRaw {
		if IsExcludedModel(m, excludes) {
			if strings.EqualFold(m.ModelID, trimmed) ||
				strings.EqualFold(m.ModelID, noLatest) ||
				strings.EqualFold(m.Name, noLatest) {
				return AvailableModel{}, fmt.Errorf("%w: %s", ErrModelExcluded, m.ModelID)
			}
		}
	}
	for _, ex := range excludes {
		pat := strings.ToLower(strings.TrimSpace(ex))
		if pat != "" && (low == pat || strings.Contains(low, pat)) {
			return AvailableModel{}, fmt.Errorf("%w: %s", ErrModelExcluded, req)
		}
	}

	// 1. Exact ID match.
	for _, m := range catalog {
		if m.ModelID == trimmed || m.ModelID == noLatest {
			return m, nil
		}
	}

	// 2. Case-insensitive ID or display name match.
	for _, m := range catalog {
		if strings.EqualFold(m.ModelID, noLatest) || strings.EqualFold(m.Name, noLatest) {
			return m, nil
		}
	}

	// 3. Base-name fallback (e.g. "claude-sonnet-5-5" -> "claude-sonnet-5-5-medium").
	for _, suffix := range []string{"-medium", "-low", "-high"} {
		candidate := low + suffix
		for _, m := range catalog {
			if strings.EqualFold(m.ModelID, candidate) {
				return m, nil
			}
		}
	}

	// 4. Short convenience aliases for general clients.
	aliasMap := map[string]string{
		"claude":        "claude-sonnet-5-5-medium",
		"claude-sonnet": "claude-sonnet-5-5-medium",
		"claude-opus":   "claude-opus-5-5-medium",
		"gemini":        "gemini-4-argon-medium",
		"gemini-flash":  "gemini-3.8-flash-medium",
		"gpt-oss":       "gpt-oss-120b-medium",
	}
	if mapped, ok := aliasMap[low]; ok {
		for _, m := range catalog {
			if strings.EqualFold(m.ModelID, mapped) {
				return m, nil
			}
		}
	}

	return AvailableModel{}, fmt.Errorf("%w: %s", ErrModelNotFound, req)
}

// Execute acquires an idle worker, dispatches the prompt request, and streams events.
// When the primary model's usage/quota is exhausted before any output is emitted,
// Execute automatically falls back to cfg.FallbackModel (default: gemini-3.8-flash-low).
func (p *Pool) Execute(ctx context.Context, req PromptRequest) (<-chan StreamEvent, error) {
	effectiveReq := req
	fallback := strings.TrimSpace(p.cfg.FallbackModel)
	if fallback != "" && effectiveReq.ModelID != fallback && p.isModelExhausted(effectiveReq.ModelID) {
		p.logger.Warn("primary model in quota cooldown; routing directly to fallback",
			"from", effectiveReq.ModelID,
			"to", fallback,
		)
		effectiveReq.ModelID = fallback
	}

	primaryCh, err := p.dispatchOnce(ctx, effectiveReq)
	if err != nil {
		return nil, err
	}

	if fallback == "" || effectiveReq.ModelID == fallback {
		return primaryCh, nil
	}

	out := make(chan StreamEvent, 64)
	go func() {
		defer close(out)
		emittedOutput := false
		for ev := range primaryCh {
			switch ev.Type {
			case EventTextChunk, EventThoughtChunk:
				emittedOutput = true
				select {
				case out <- ev:
				case <-ctx.Done():
					return
				}
			case EventError:
				if !emittedOutput && errors.Is(ev.Err, ErrQuotaExhausted) && ctx.Err() == nil {
					p.markModelExhausted(effectiveReq.ModelID)
					p.logger.Warn("primary model usage/quota exhausted; failing over to fallback",
						"from", effectiveReq.ModelID,
						"to", fallback,
						"err", ev.Err,
					)
					fbReq := effectiveReq
					fbReq.ModelID = fallback
					fbCh, fbErr := p.dispatchOnce(ctx, fbReq)
					if fbErr != nil {
						select {
						case out <- StreamEvent{Type: EventError, Err: fbErr}:
						case <-ctx.Done():
						}
						return
					}
					for fbEv := range fbCh {
						select {
						case out <- fbEv:
						case <-ctx.Done():
							return
						}
					}
					return
				}
				select {
				case out <- ev:
				case <-ctx.Done():
				}
				return
			default:
				select {
				case out <- ev:
				case <-ctx.Done():
					return
				}
			}
		}
	}()
	return out, nil
}

func (p *Pool) dispatchOnce(ctx context.Context, req PromptRequest) (<-chan StreamEvent, error) {
	var w *worker
	select {
	case w = <-p.idle:
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-p.closedCh:
		return nil, ErrPoolClosed
	}

	job := &workerJob{
		ctx:    ctx,
		req:    req,
		events: make(chan StreamEvent, 64),
	}

	select {
	case w.jobs <- job:
		return job.events, nil
	case <-ctx.Done():
		p.release(w)
		return nil, ctx.Err()
	case <-p.closedCh:
		p.release(w)
		return nil, ErrPoolClosed
	}
}

func (p *Pool) isModelExhausted(modelID string) bool {
	if p.cfg.FallbackCooldown <= 0 {
		return false
	}
	p.mu.RLock()
	until, ok := p.exhaustedUntil[modelID]
	p.mu.RUnlock()
	return ok && time.Now().Before(until)
}

func (p *Pool) markModelExhausted(modelID string) {
	if p.cfg.FallbackCooldown <= 0 || modelID == "" {
		return
	}
	p.mu.Lock()
	p.exhaustedUntil[modelID] = time.Now().Add(p.cfg.FallbackCooldown)
	p.mu.Unlock()
}
