package ollama

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"ollama-antigravity/internal/acp"
)

const (
	ollamaCompatVersion = "0.40.16"
	maxRequestBodyBytes = 32 * 1024 * 1024
)

// Server implements the Ollama-compatible HTTP API backed by an acp.Executor.
type Server struct {
	exec      acp.Executor
	logger    *slog.Logger
	startedAt time.Time
}

// NewServer constructs an HTTP server handler for Ollama API routes.
func NewServer(exec acp.Executor, logger *slog.Logger) *Server {
	return &Server{
		exec:      exec,
		logger:    logger,
		startedAt: time.Now().UTC(),
	}
}

// Handler returns the root http.Handler with routing and middleware applied.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleRoot)
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /api/version", s.handleVersion)
	mux.HandleFunc("GET /api/tags", s.handleTags)
	mux.HandleFunc("GET /api/ps", s.handlePS)
	mux.HandleFunc("POST /api/show", s.handleShow)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/generate", s.handleGenerate)

	return s.withCORS(s.withLogging(mux))
}

func (s *Server) withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		s.logger.Debug("http request",
			"method", r.Method,
			"path", r.URL.Path,
			"remote", r.RemoteAddr,
			"duration_ms", time.Since(start).Milliseconds(),
		)
	})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if r.Method != http.MethodHead {
		_, _ = io.WriteString(w, "Ollama is running")
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	models := s.exec.ListModels()
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"models": len(models),
	})
}

func (s *Server) handleVersion(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{
		"version": ollamaCompatVersion,
	})
}

func (s *Server) handlePS(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"models": []any{},
	})
}

func (s *Server) handleTags(w http.ResponseWriter, _ *http.Request) {
	avail := s.exec.ListModels()
	out := make([]TagModel, 0, len(avail))
	modTime := s.startedAt.Format(time.RFC3339Nano)

	for _, m := range avail {
		family, _ := modelMetadata(m.ModelID)
		out = append(out, TagModel{
			Name:       m.ModelID,
			Model:      m.ModelID,
			ModifiedAt: modTime,
			Size:       0,
			Digest:     modelDigest(m.ModelID),
			Details: ModelDetails{
				ParentModel:       "",
				Format:            "acp",
				Family:            family,
				Families:          []string{family},
				ParameterSize:     "cloud",
				QuantizationLevel: "cloud",
			},
		})
	}

	writeJSON(w, http.StatusOK, TagsResponse{Models: out})
}

func (s *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	var req ShowRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	name := strings.TrimSpace(req.Model)
	if name == "" {
		name = strings.TrimSpace(req.Name)
	}
	if name == "" {
		writeError(w, http.StatusBadRequest, "model is required")
		return
	}

	resolved, err := s.exec.ResolveModel(name)
	if err != nil {
		writeModelError(w, err)
		return
	}

	family, ctxLen := modelMetadata(resolved.ModelID)
	writeJSON(w, http.StatusOK, ShowResponse{
		License:    "Antigravity ACP Bridge",
		Modelfile:  fmt.Sprintf("FROM %s", resolved.ModelID),
		Parameters: "",
		Template:   "{{ .Prompt }}",
		Details: ModelDetails{
			ParentModel:       "",
			Format:            "acp",
			Family:            family,
			Families:          []string{family},
			ParameterSize:     "cloud",
			QuantizationLevel: "cloud",
		},
		ModelInfo: map[string]any{
			"general.architecture":               family,
			fmt.Sprintf("%s.context_length", family): ctxLen,
		},
		Capabilities: []string{"completion", "vision", "tools"},
		ModifiedAt:   s.startedAt.Format(time.RFC3339Nano),
	})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages cannot be empty")
		return
	}

	resolved, err := s.exec.ResolveModel(req.Model)
	if err != nil {
		writeModelError(w, err)
		return
	}

	displayModel := strings.TrimSpace(req.Model)
	if displayModel == "" {
		displayModel = resolved.ModelID
	}

	stream := req.Stream == nil || *req.Stream
	includeThink := req.Think != nil && *req.Think

	promptReq := buildChatPrompt(resolved.ModelID, req)
	events, err := s.exec.Execute(r.Context(), promptReq)
	if err != nil {
		s.logger.Error("acp execute failed", "model", resolved.ModelID, "err", err)
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	hasFormat := len(strings.TrimSpace(string(req.Format))) > 0 && string(req.Format) != "null"
	hasTools := len(req.Tools) > 0

	// Buffer when non-streaming OR when tools are supplied (so tool_calls JSON isn't leaked as text chunks).
	if !stream || hasTools {
		res, err := collectEvents(events, includeThink)
		if err != nil {
			s.logger.Error("acp stream failed", "model", resolved.ModelID, "err", err)
			writeExecError(w, err)
			return
		}

		content := res.text
		var toolCalls []ToolCall
		if hasTools {
			if parsed, rem, ok := parseToolCalls(content); ok {
				toolCalls = parsed
				content = rem
			}
		}
		if hasFormat && len(toolCalls) == 0 {
			content = stripJSONFencesIfValid(content)
		}

		msg := ChatMessage{
			Role:      "assistant",
			Content:   content,
			Thinking:  res.thinking,
			ToolCalls: toolCalls,
		}

		final := buildFinalChatResponse(displayModel, msg, res.done)
		if stream {
			writeSingleNDJSON(w, final)
			return
		}
		writeJSON(w, http.StatusOK, final)
		return
	}

	s.streamChatNDJSON(w, displayModel, events, includeThink)
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req GenerateRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	resolved, err := s.exec.ResolveModel(req.Model)
	if err != nil {
		writeModelError(w, err)
		return
	}

	displayModel := strings.TrimSpace(req.Model)
	if displayModel == "" {
		displayModel = resolved.ModelID
	}

	stream := req.Stream == nil || *req.Stream
	includeThink := req.Think != nil && *req.Think

	var msgs []ChatMessage
	if strings.TrimSpace(req.System) != "" {
		msgs = append(msgs, ChatMessage{Role: "system", Content: req.System})
	}
	msgs = append(msgs, ChatMessage{Role: "user", Content: req.Prompt, Images: req.Images})

	promptReq := buildChatPrompt(resolved.ModelID, ChatRequest{
		Model:    req.Model,
		Messages: msgs,
		Format:   req.Format,
		Options:  req.Options,
	})

	events, err := s.exec.Execute(r.Context(), promptReq)
	if err != nil {
		s.logger.Error("acp execute failed", "model", resolved.ModelID, "err", err)
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	hasFormat := len(strings.TrimSpace(string(req.Format))) > 0 && string(req.Format) != "null"

	if !stream {
		res, err := collectEvents(events, includeThink)
		if err != nil {
			s.logger.Error("acp stream failed", "model", resolved.ModelID, "err", err)
			writeExecError(w, err)
			return
		}
		text := res.text
		if hasFormat {
			text = stripJSONFencesIfValid(text)
		}
		writeJSON(w, http.StatusOK, buildFinalGenerateResponse(displayModel, text, res.thinking, res.done))
		return
	}

	s.streamGenerateNDJSON(w, displayModel, events, includeThink)
}

type collectedResult struct {
	text     string
	thinking string
	done     acp.StreamEvent
}

func collectEvents(events <-chan acp.StreamEvent, includeThink bool) (collectedResult, error) {
	var textBuf strings.Builder
	var thinkBuf strings.Builder
	var doneEv acp.StreamEvent

	for ev := range events {
		switch ev.Type {
		case acp.EventTextChunk:
			textBuf.WriteString(ev.Text)
		case acp.EventThoughtChunk:
			if includeThink {
				thinkBuf.WriteString(ev.Text)
			}
		case acp.EventDone:
			doneEv = ev
		case acp.EventError:
			return collectedResult{}, ev.Err
		}
	}

	return collectedResult{
		text:     textBuf.String(),
		thinking: thinkBuf.String(),
		done:     doneEv,
	}, nil
}

func (s *Server) streamChatNDJSON(w http.ResponseWriter, model string, events <-chan acp.StreamEvent, includeThink bool) {
	flusher, ok := prepareNDJSON(w)
	if !ok {
		return
	}

	for ev := range events {
		switch ev.Type {
		case acp.EventTextChunk:
			_ = writeNDJSONLine(w, flusher, ChatResponse{
				Model:     model,
				CreatedAt: nowRFC3339(),
				Message: ChatMessage{
					Role:    "assistant",
					Content: ev.Text,
				},
				Done: false,
			})
		case acp.EventThoughtChunk:
			if includeThink {
				_ = writeNDJSONLine(w, flusher, ChatResponse{
					Model:     model,
					CreatedAt: nowRFC3339(),
					Message: ChatMessage{
						Role:     "assistant",
						Thinking: ev.Text,
					},
					Done: false,
				})
			}
		case acp.EventDone:
			_ = writeNDJSONLine(w, flusher, buildFinalChatResponse(model, ChatMessage{Role: "assistant"}, ev))
			return
		case acp.EventError:
			_ = writeNDJSONLine(w, flusher, ErrorResponse{Error: ev.Err.Error()})
			return
		}
	}
}

func (s *Server) streamGenerateNDJSON(w http.ResponseWriter, model string, events <-chan acp.StreamEvent, includeThink bool) {
	flusher, ok := prepareNDJSON(w)
	if !ok {
		return
	}

	for ev := range events {
		switch ev.Type {
		case acp.EventTextChunk:
			_ = writeNDJSONLine(w, flusher, GenerateResponse{
				Model:     model,
				CreatedAt: nowRFC3339(),
				Response:  ev.Text,
				Done:      false,
			})
		case acp.EventThoughtChunk:
			if includeThink {
				_ = writeNDJSONLine(w, flusher, GenerateResponse{
					Model:     model,
					CreatedAt: nowRFC3339(),
					Thinking:  ev.Text,
					Done:      false,
				})
			}
		case acp.EventDone:
			_ = writeNDJSONLine(w, flusher, buildFinalGenerateResponse(model, "", "", ev))
			return
		case acp.EventError:
			_ = writeNDJSONLine(w, flusher, ErrorResponse{Error: ev.Err.Error()})
			return
		}
	}
}

func buildFinalChatResponse(model string, msg ChatMessage, done acp.StreamEvent) ChatResponse {
	totalDur := done.LoadDur + done.EvalDur
	return ChatResponse{
		Model:              model,
		CreatedAt:          nowRFC3339(),
		Message:            msg,
		Done:               true,
		DoneReason:         "stop",
		TotalDuration:      totalDur.Nanoseconds(),
		LoadDuration:       done.LoadDur.Nanoseconds(),
		PromptEvalCount:    done.Usage.InputTokens,
		PromptEvalDuration: done.LoadDur.Nanoseconds(),
		EvalCount:          done.Usage.OutputTokens,
		EvalDuration:       done.EvalDur.Nanoseconds(),
	}
}

func buildFinalGenerateResponse(model, text, thinking string, done acp.StreamEvent) GenerateResponse {
	totalDur := done.LoadDur + done.EvalDur
	return GenerateResponse{
		Model:              model,
		CreatedAt:          nowRFC3339(),
		Response:           text,
		Thinking:           thinking,
		Done:               true,
		DoneReason:         "stop",
		TotalDuration:      totalDur.Nanoseconds(),
		LoadDuration:       done.LoadDur.Nanoseconds(),
		PromptEvalCount:    done.Usage.InputTokens,
		PromptEvalDuration: done.LoadDur.Nanoseconds(),
		EvalCount:          done.Usage.OutputTokens,
		EvalDuration:       done.EvalDur.Nanoseconds(),
	}
}

func buildChatPrompt(modelID string, req ChatRequest) acp.PromptRequest {
	sysParts := []string{
		"You are operating in HEADLESS INFERENCE MODE behind an Ollama-compatible API bridge. " +
			"In this mode, all Antigravity native function-calling and tool-use channels " +
			"(run_command, write_to_file, view_file, read_file, edit_file, replace_file_content, " +
			"multi_replace_file_content, find_by_name, grep_search, list_dir, browser, etc.) are DISABLED " +
			"and will fail or terminate the session immediately. NEVER emit native Antigravity function calls " +
			"or invoke local filesystem, shell, or IDE tools. Answer directly from the conversation input.",
	}

	var convMsgs []ChatMessage
	for _, m := range req.Messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		if role == "system" || role == "developer" {
			if strings.TrimSpace(m.Content) != "" {
				sysParts = append(sysParts, m.Content)
			}
			continue
		}
		convMsgs = append(convMsgs, m)
	}

	if formatHint := formatInstruction(req.Format); formatHint != "" {
		sysParts = append(sysParts, formatHint)
	}
	if toolHint := toolsInstruction(req.Tools); toolHint != "" {
		sysParts = append(sysParts, toolHint)
	}
	if optHint := optionsInstruction(req.Options); optHint != "" {
		sysParts = append(sysParts, optHint)
	}

	var blocks []acp.ContentBlock
	if len(convMsgs) == 1 && strings.EqualFold(strings.TrimSpace(convMsgs[0].Role), "user") && len(req.Tools) == 0 {
		blocks = append(blocks, acp.ContentBlock{Type: "text", Text: convMsgs[0].Content})
		for _, img := range convMsgs[0].Images {
			blocks = append(blocks, imageBlock(img))
		}
	} else {
		var transcript strings.Builder
		transcript.WriteString("Continue the conversation below by producing only the next assistant reply.\n\n")
		lastRole := ""
		for _, m := range convMsgs {
			role := strings.ToLower(strings.TrimSpace(m.Role))
			lastRole = role
			switch role {
			case "tool":
				toolLabel := strings.TrimSpace(m.ToolName)
				if toolLabel == "" {
					toolLabel = "tool"
				}
				transcript.WriteString("Tool Result (")
				transcript.WriteString(toolLabel)
				transcript.WriteString("):\n")
				transcript.WriteString(m.Content)
				transcript.WriteString("\n\n")
			case "assistant":
				transcript.WriteString("[ASSISTANT]\n")
				if m.Content != "" {
					transcript.WriteString(m.Content)
					transcript.WriteString("\n")
				}
				for i, tc := range m.ToolCalls {
					callObj := map[string]any{
						"id":   fmt.Sprintf("call_%d", i+1),
						"type": "function",
						"function": map[string]any{
							"name":      tc.Function.Name,
							"arguments": tc.Function.Arguments,
						},
					}
					if tcJSON, err := json.Marshal(callObj); err == nil {
						transcript.WriteString("<tool_call>")
						transcript.Write(tcJSON)
						transcript.WriteString("</tool_call>\n")
					}
				}
				transcript.WriteString("\n")
			default:
				transcript.WriteString("[USER]\n")
				if m.Content != "" {
					transcript.WriteString(m.Content)
					transcript.WriteString("\n")
				}
				transcript.WriteString("\n")
			}
			for _, img := range m.Images {
				blocks = append(blocks, imageBlock(img))
			}
		}
		if lastRole == "tool" {
			transcript.WriteString("REMINDER: You are in HEADLESS INFERENCE MODE. Native Antigravity tools are disabled. " +
				"If more client tools are needed, emit ONLY <tool_call>{...}</tool_call> tags; otherwise reply normally.\n\n")
		}
		transcript.WriteString("[ASSISTANT]\n")
		blocks = append([]acp.ContentBlock{{Type: "text", Text: transcript.String()}}, blocks...)
	}

	return acp.PromptRequest{
		ModelID:      modelID,
		SystemPrompt: strings.Join(sysParts, "\n\n"),
		Prompt:       blocks,
	}
}

func formatInstruction(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	var strVal string
	if err := json.Unmarshal(raw, &strVal); err == nil {
		if strings.EqualFold(strings.TrimSpace(strVal), "json") {
			return "OUTPUT CONSTRAINT: Respond with valid raw JSON only. Do not include markdown code fences, prose, or explanations."
		}
		return ""
	}
	return "OUTPUT CONSTRAINT: Respond with a single valid raw JSON value strictly conforming to the following JSON Schema. " +
		"Do not include markdown code fences, prose, or explanations:\n" + trimmed
}

func toolsInstruction(tools []ToolDef) string {
	if len(tools) == 0 {
		return ""
	}
	b, err := json.Marshal(tools)
	if err != nil {
		return ""
	}
	return "AVAILABLE CLIENT TOOLS (JSON):\n" + string(b) + "\n\n" +
		"IMPORTANT INSTRUCTIONS FOR CLIENT TOOLS:\n" +
		"- If you need to call a client tool, emit ONLY <tool_call>{...}</tool_call> blocks in plain text — NEVER invoke native Antigravity tools.\n" +
		`- Each <tool_call> block must contain a valid JSON object: {"id":"call_1","type":"function","function":{"name":"<tool_name>","arguments":{...}}}.` + "\n" +
		"- NEVER simulate or hallucinate Tool Results. After emitting <tool_call> blocks, stop generating immediately.\n" +
		"- If no tool is needed, respond normally with plain text."
}

func optionsInstruction(opts map[string]any) string {
	if len(opts) == 0 {
		return ""
	}
	var hints []string
	if stopRaw, ok := opts["stop"]; ok {
		if b, err := json.Marshal(stopRaw); err == nil {
			hints = append(hints, "stop sequences: "+string(b))
		}
	}
	if np, ok := opts["num_predict"]; ok {
		hints = append(hints, fmt.Sprintf("max output tokens target: %v", np))
	}
	if len(hints) == 0 {
		return ""
	}
	return "GENERATION HINTS: " + strings.Join(hints, "; ")
}

func imageBlock(b64 string) acp.ContentBlock {
	clean := strings.TrimSpace(b64)
	mimeType := "image/png"
	if strings.HasPrefix(clean, "data:") {
		if comma := strings.IndexByte(clean, ','); comma > 5 {
			header := clean[5:comma]
			if semi := strings.IndexByte(header, ';'); semi > 0 {
				mimeType = header[:semi]
			} else if header != "" {
				mimeType = header
			}
			clean = clean[comma+1:]
		}
	} else {
		sample := clean
		if len(sample) > 32 {
			sample = sample[:32]
		}
		if raw, err := base64.StdEncoding.DecodeString(sample); err == nil && len(raw) >= 4 {
			mimeType = sniffImageMime(raw)
		}
	}
	return acp.ContentBlock{
		Type:     "image",
		Data:     clean,
		MimeType: mimeType,
	}
}

func sniffImageMime(b []byte) string {
	if len(b) >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF {
		return "image/jpeg"
	}
	if len(b) >= 4 && b[0] == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G' {
		return "image/png"
	}
	if len(b) >= 3 && b[0] == 'G' && b[1] == 'I' && b[2] == 'F' {
		return "image/gif"
	}
	if len(b) >= 12 && string(b[0:4]) == "RIFF" && string(b[8:12]) == "WEBP" {
		return "image/webp"
	}
	return "image/png"
}

const (
	toolCallOpenTag  = "<tool_call>"
	toolCallCloseTag = "</tool_call>"
)

// parseToolCalls extracts client tool calls from either <tool_call>{...}</tool_call>
// blocks (Hermes directsdk protocol) or a raw {"tool_calls":[...]} JSON envelope.
func parseToolCalls(raw string) ([]ToolCall, string, bool) {
	if calls, rem, ok := parseToolCallTags(raw); ok {
		return calls, rem, true
	}
	if calls, ok := parseToolCallsEnvelope(raw); ok {
		return calls, "", true
	}
	return nil, raw, false
}

func parseToolCallTags(raw string) ([]ToolCall, string, bool) {
	var calls []ToolCall
	var textOut strings.Builder
	cursor := 0

	for cursor < len(raw) {
		rel := strings.Index(raw[cursor:], toolCallOpenTag)
		if rel < 0 {
			textOut.WriteString(raw[cursor:])
			break
		}
		openIdx := cursor + rel
		textOut.WriteString(raw[cursor:openIdx])

		bodyStart := openIdx + len(toolCallOpenTag)
		closeIdx := findToolCallClose(raw, bodyStart)
		if closeIdx < 0 {
			return nil, raw, false
		}

		body := strings.TrimSpace(raw[bodyStart:closeIdx])
		tc, ok := decodeSingleToolCall(body, len(calls))
		if !ok {
			return nil, raw, false
		}
		calls = append(calls, tc)
		cursor = closeIdx + len(toolCallCloseTag)
	}

	if len(calls) == 0 {
		return nil, raw, false
	}
	return calls, strings.TrimSpace(textOut.String()), true
}

// findToolCallClose locates </tool_call> outside JSON string literals so
// embedded closing tags inside tool arguments do not prematurely split a block.
func findToolCallClose(s string, start int) int {
	inString := false
	escaped := false
	for i := start; i < len(s); i++ {
		ch := s[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '"' {
			inString = true
			continue
		}
		if strings.HasPrefix(s[i:], toolCallCloseTag) {
			return i
		}
	}
	return -1
}

func decodeSingleToolCall(body string, index int) (ToolCall, bool) {
	var rawObj struct {
		Function *struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(body), &rawObj); err != nil {
		return ToolCall{}, false
	}

	name := strings.TrimSpace(rawObj.Name)
	argsRaw := rawObj.Arguments
	if rawObj.Function != nil {
		name = strings.TrimSpace(rawObj.Function.Name)
		argsRaw = rawObj.Function.Arguments
	}
	if name == "" {
		return ToolCall{}, false
	}

	args := parseArgumentsMap(argsRaw)
	return ToolCall{
		Function: ToolCallFunction{
			Index:     index,
			Name:      name,
			Arguments: args,
		},
	}, true
}

func parseArgumentsMap(raw json.RawMessage) map[string]any {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return map[string]any{}
	}
	var asMap map[string]any
	if err := json.Unmarshal(raw, &asMap); err == nil && asMap != nil {
		return asMap
	}
	var asStr string
	if err := json.Unmarshal(raw, &asStr); err == nil {
		var nested map[string]any
		if err := json.Unmarshal([]byte(asStr), &nested); err == nil && nested != nil {
			return nested
		}
	}
	return map[string]any{}
}

func parseToolCallsEnvelope(raw string) ([]ToolCall, bool) {
	cleaned := stripMarkdownFence(raw)
	if !strings.HasPrefix(cleaned, "{") {
		return nil, false
	}
	var env struct {
		ToolCalls []ToolCall `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(cleaned), &env); err != nil || len(env.ToolCalls) == 0 {
		return nil, false
	}
	for i := range env.ToolCalls {
		if strings.TrimSpace(env.ToolCalls[i].Function.Name) == "" {
			return nil, false
		}
		if env.ToolCalls[i].Function.Arguments == nil {
			env.ToolCalls[i].Function.Arguments = map[string]any{}
		}
		env.ToolCalls[i].Function.Index = i
	}
	return env.ToolCalls, true
}

func stripJSONFencesIfValid(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if json.Valid([]byte(trimmed)) {
		return trimmed
	}
	unwrapped := stripMarkdownFence(trimmed)
	if json.Valid([]byte(unwrapped)) {
		return unwrapped
	}
	return raw
}

func stripMarkdownFence(s string) string {
	trimmed := strings.TrimSpace(s)
	if !strings.HasPrefix(trimmed, "```") {
		return trimmed
	}
	nl := strings.IndexByte(trimmed, '\n')
	if nl < 0 {
		return trimmed
	}
	body := trimmed[nl+1:]
	if idx := strings.LastIndex(body, "```"); idx >= 0 {
		body = body[:idx]
	}
	return strings.TrimSpace(body)
}

func modelMetadata(modelID string) (family string, contextLength int) {
	low := strings.ToLower(modelID)
	switch {
	case strings.Contains(low, "claude"):
		return "claude", 200000
	case strings.Contains(low, "gemini"):
		return "gemini", 1048576
	case strings.Contains(low, "gpt"):
		return "gpt", 131072
	default:
		return "acp", 131072
	}
}

func modelDigest(modelID string) string {
	sum := sha256.Sum256([]byte("acp:" + modelID))
	return hex.EncodeToString(sum[:])
}

func decodeJSONBody(r *http.Request, dst any) error {
	defer r.Body.Close()
	limited := io.LimitReader(r.Body, maxRequestBodyBytes)
	dec := json.NewDecoder(limited)
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("invalid JSON body: %w", err)
	}
	return nil
}

func writeExecError(w http.ResponseWriter, err error) {
	if errors.Is(err, acp.ErrNativeToolBlocked) {
		writeError(w, http.StatusForbidden, err.Error())
		return
	}
	writeError(w, http.StatusBadGateway, err.Error())
}

func writeModelError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, acp.ErrModelExcluded):
		writeError(w, http.StatusNotFound, err.Error())
	case errors.Is(err, acp.ErrModelNotFound):
		writeError(w, http.StatusNotFound, err.Error())
	default:
		writeError(w, http.StatusInternalServerError, err.Error())
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, ErrorResponse{Error: msg})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func prepareNDJSON(w http.ResponseWriter) (http.Flusher, bool) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return nil, false
	}
	w.WriteHeader(http.StatusOK)
	return flusher, true
}

func writeSingleNDJSON(w http.ResponseWriter, v any) {
	flusher, ok := prepareNDJSON(w)
	if !ok {
		return
	}
	_ = writeNDJSONLine(w, flusher, v)
}

func writeNDJSONLine(w io.Writer, flusher http.Flusher, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if _, err := w.Write(b); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func nowRFC3339() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}
