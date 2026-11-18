package ollama

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"ollama-antigravity/internal/acp"
)

var completionSeq atomic.Uint64

type openAIModelsList struct {
	Object string            `json:"object"`
	Data   []openAIModelItem `json:"data"`
}

type openAIModelItem struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type openAIErrorEnvelope struct {
	Error openAIErrorDetail `json:"error"`
}

type openAIErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

type openAIChatRequest struct {
	Model               string          `json:"model"`
	Messages            []openAIMessage `json:"messages"`
	Tools               []ToolDef       `json:"tools,omitempty"`
	ResponseFormat      json.RawMessage `json:"response_format,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	StreamOptions       *streamOptions  `json:"stream_options,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type openAIMessage struct {
	Role             string           `json:"role"`
	Content          json.RawMessage  `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	Name             string           `json:"name,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIToolCall struct {
	Index    *int               `json:"index,omitempty"`
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIToolFunction `json:"function"`
}

type openAIToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIChatResponse struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []openAIChatChoice   `json:"choices"`
	Usage   *openAIUsage         `json:"usage,omitempty"`
}

type openAIChatChoice struct {
	Index        int                  `json:"index"`
	Message      *openAIRespMessage   `json:"message,omitempty"`
	Delta        *openAIRespDelta     `json:"delta,omitempty"`
	FinishReason *string              `json:"finish_reason"`
}

type openAIRespMessage struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIRespDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAICompletionRequest struct {
	Model     string          `json:"model"`
	Prompt    json.RawMessage `json:"prompt"`
	Stream    bool            `json:"stream,omitempty"`
	MaxTokens *int            `json:"max_tokens,omitempty"`
	Stop      json.RawMessage `json:"stop,omitempty"`
}

type openAICompletionResponse struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []openAICompletionChoice  `json:"choices"`
	Usage   *openAIUsage              `json:"usage,omitempty"`
}

type openAICompletionChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

func (s *Server) handleV1Models(w http.ResponseWriter, _ *http.Request) {
	avail := s.exec.ListModels()
	created := s.startedAt.Unix()
	data := make([]openAIModelItem, 0, len(avail))
	for _, m := range avail {
		data = append(data, openAIModelItem{
			ID:      m.ModelID,
			Object:  "model",
			Created: created,
			OwnedBy: "antigravity-acp",
		})
	}
	writeJSON(w, http.StatusOK, openAIModelsList{
		Object: "list",
		Data:   data,
	})
}

func (s *Server) handleV1ModelGet(w http.ResponseWriter, r *http.Request) {
	modelID := strings.TrimSpace(r.PathValue("model"))
	if modelID == "" {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "model_required", "model is required")
		return
	}
	resolved, err := s.exec.ResolveModel(modelID)
	if err != nil {
		writeOpenAIModelError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, openAIModelItem{
		ID:      resolved.ModelID,
		Object:  "model",
		Created: s.startedAt.Unix(),
		OwnedBy: "antigravity-acp",
	})
}

func (s *Server) handleV1ChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req openAIChatRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "empty_messages", "messages cannot be empty")
		return
	}

	resolved, err := s.exec.ResolveModel(req.Model)
	if err != nil {
		writeOpenAIModelError(w, err)
		return
	}

	displayModel := strings.TrimSpace(req.Model)
	if displayModel == "" {
		displayModel = resolved.ModelID
	}

	chatReq, err := convertOpenAIChatToOllama(req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_messages", err.Error())
		return
	}

	promptReq := buildChatPrompt(resolved.ModelID, chatReq)
	events, err := s.exec.Execute(r.Context(), promptReq)
	if err != nil {
		s.logger.Error("acp execute failed (v1)", "model", resolved.ModelID, "err", err)
		writeOpenAIExecError(w, err)
		return
	}

	compID := fmt.Sprintf("chatcmpl-%d", completionSeq.Add(1))
	created := time.Now().UTC().Unix()
	hasTools := len(req.Tools) > 0
	hasFormat := len(chatReq.Format) > 0

	if !req.Stream || hasTools {
		res, err := collectEvents(events, true)
		if err != nil {
			s.logger.Error("acp stream failed (v1)", "model", resolved.ModelID, "err", err)
			writeOpenAIExecError(w, err)
			return
		}

		content := res.text
		finishReason := "stop"
		var outToolCalls []openAIToolCall

		if hasTools {
			if parsed, rem, ok := parseToolCalls(content); ok {
				finishReason = "tool_calls"
				content = rem
				for i, tc := range parsed {
					idx := i
					argsBytes, _ := json.Marshal(tc.Function.Arguments)
					outToolCalls = append(outToolCalls, openAIToolCall{
						Index: &idx,
						ID:    fmt.Sprintf("call_%d_%d", created, i+1),
						Type:  "function",
						Function: openAIToolFunction{
							Name:      tc.Function.Name,
							Arguments: string(argsBytes),
						},
					})
				}
			}
		}
		if hasFormat && len(outToolCalls) == 0 {
			content = stripJSONFencesIfValid(content)
		}

		usage := &openAIUsage{
			PromptTokens:     res.done.Usage.InputTokens,
			CompletionTokens: res.done.Usage.OutputTokens,
			TotalTokens:      res.done.Usage.InputTokens + res.done.Usage.OutputTokens,
		}

		if req.Stream {
			s.writeBufferedOpenAISSE(w, compID, created, displayModel, content, res.thinking, outToolCalls, finishReason, usage)
			return
		}

		writeJSON(w, http.StatusOK, openAIChatResponse{
			ID:      compID,
			Object:  "chat.completion",
			Created: created,
			Model:   displayModel,
			Choices: []openAIChatChoice{
				{
					Index: 0,
					Message: &openAIRespMessage{
						Role:             "assistant",
						Content:          content,
						ReasoningContent: res.thinking,
						ToolCalls:        outToolCalls,
					},
					FinishReason: &finishReason,
				},
			},
			Usage: usage,
		})
		return
	}

	s.streamOpenAIChatSSE(w, compID, created, displayModel, events, req.StreamOptions != nil && req.StreamOptions.IncludeUsage)
}

func (s *Server) handleV1Completions(w http.ResponseWriter, r *http.Request) {
	var req openAICompletionRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}

	resolved, err := s.exec.ResolveModel(req.Model)
	if err != nil {
		writeOpenAIModelError(w, err)
		return
	}

	displayModel := strings.TrimSpace(req.Model)
	if displayModel == "" {
		displayModel = resolved.ModelID
	}

	promptText := extractCompletionPrompt(req.Prompt)
	opts := map[string]any{}
	if req.MaxTokens != nil {
		opts["num_predict"] = *req.MaxTokens
	}
	if len(strings.TrimSpace(string(req.Stop))) > 0 && string(req.Stop) != "null" {
		var stopAny any
		if json.Unmarshal(req.Stop, &stopAny) == nil {
			opts["stop"] = stopAny
		}
	}

	promptReq := buildChatPrompt(resolved.ModelID, ChatRequest{
		Model:    req.Model,
		Messages: []ChatMessage{{Role: "user", Content: promptText}},
		Options:  opts,
	})

	events, err := s.exec.Execute(r.Context(), promptReq)
	if err != nil {
		writeOpenAIExecError(w, err)
		return
	}

	compID := fmt.Sprintf("cmpl-%d", completionSeq.Add(1))
	created := time.Now().UTC().Unix()

	if !req.Stream {
		res, err := collectEvents(events, false)
		if err != nil {
			writeOpenAIExecError(w, err)
			return
		}
		finish := "stop"
		writeJSON(w, http.StatusOK, openAICompletionResponse{
			ID:      compID,
			Object:  "text_completion",
			Created: created,
			Model:   displayModel,
			Choices: []openAICompletionChoice{
				{Index: 0, Text: res.text, FinishReason: &finish},
			},
			Usage: &openAIUsage{
				PromptTokens:     res.done.Usage.InputTokens,
				CompletionTokens: res.done.Usage.OutputTokens,
				TotalTokens:      res.done.Usage.InputTokens + res.done.Usage.OutputTokens,
			},
		})
		return
	}

	flusher, ok := prepareSSE(w)
	if !ok {
		return
	}
	for ev := range events {
		switch ev.Type {
		case acp.EventTextChunk:
			_ = writeSSEData(w, flusher, openAICompletionResponse{
				ID:      compID,
				Object:  "text_completion",
				Created: created,
				Model:   displayModel,
				Choices: []openAICompletionChoice{{Index: 0, Text: ev.Text}},
			})
		case acp.EventDone:
			finish := "stop"
			_ = writeSSEData(w, flusher, openAICompletionResponse{
				ID:      compID,
				Object:  "text_completion",
				Created: created,
				Model:   displayModel,
				Choices: []openAICompletionChoice{{Index: 0, Text: "", FinishReason: &finish}},
			})
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
			return
		case acp.EventError:
			_ = writeSSEData(w, flusher, openAIErrorEnvelope{
				Error: openAIErrorDetail{Message: ev.Err.Error(), Type: "server_error"},
			})
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
			return
		}
	}
}

func (s *Server) streamOpenAIChatSSE(
	w http.ResponseWriter,
	compID string,
	created int64,
	model string,
	events <-chan acp.StreamEvent,
	includeUsage bool,
) {
	flusher, ok := prepareSSE(w)
	if !ok {
		return
	}

	_ = writeSSEData(w, flusher, openAIChatResponse{
		ID:      compID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []openAIChatChoice{{
			Index: 0,
			Delta: &openAIRespDelta{Role: "assistant"},
		}},
	})

	for ev := range events {
		switch ev.Type {
		case acp.EventTextChunk:
			_ = writeSSEData(w, flusher, openAIChatResponse{
				ID:      compID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openAIChatChoice{{
					Index: 0,
					Delta: &openAIRespDelta{Content: ev.Text},
				}},
			})
		case acp.EventThoughtChunk:
			_ = writeSSEData(w, flusher, openAIChatResponse{
				ID:      compID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openAIChatChoice{{
					Index: 0,
					Delta: &openAIRespDelta{ReasoningContent: ev.Text},
				}},
			})
		case acp.EventDone:
			finish := "stop"
			chunk := openAIChatResponse{
				ID:      compID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []openAIChatChoice{{
					Index:        0,
					Delta:        &openAIRespDelta{},
					FinishReason: &finish,
				}},
			}
			if includeUsage {
				chunk.Usage = &openAIUsage{
					PromptTokens:     ev.Usage.InputTokens,
					CompletionTokens: ev.Usage.OutputTokens,
					TotalTokens:      ev.Usage.InputTokens + ev.Usage.OutputTokens,
				}
			}
			_ = writeSSEData(w, flusher, chunk)
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
			return
		case acp.EventError:
			_ = writeSSEData(w, flusher, openAIErrorEnvelope{
				Error: openAIErrorDetail{Message: ev.Err.Error(), Type: "server_error"},
			})
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			flusher.Flush()
			return
		}
	}
}

func (s *Server) writeBufferedOpenAISSE(
	w http.ResponseWriter,
	compID string,
	created int64,
	model string,
	content string,
	thinking string,
	toolCalls []openAIToolCall,
	finishReason string,
	usage *openAIUsage,
) {
	flusher, ok := prepareSSE(w)
	if !ok {
		return
	}

	_ = writeSSEData(w, flusher, openAIChatResponse{
		ID:      compID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []openAIChatChoice{{
			Index: 0,
			Delta: &openAIRespDelta{
				Role:             "assistant",
				Content:          content,
				ReasoningContent: thinking,
				ToolCalls:        toolCalls,
			},
		}},
	})
	_ = writeSSEData(w, flusher, openAIChatResponse{
		ID:      compID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []openAIChatChoice{{
			Index:        0,
			Delta:        &openAIRespDelta{},
			FinishReason: &finishReason,
		}},
		Usage: usage,
	})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

func convertOpenAIChatToOllama(req openAIChatRequest) (ChatRequest, error) {
	msgs := make([]ChatMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		text, images, err := decodeOpenAIContent(m.Content)
		if err != nil {
			return ChatRequest{}, err
		}
		var tcs []ToolCall
		for i, tc := range m.ToolCalls {
			tcs = append(tcs, ToolCall{
				Function: ToolCallFunction{
					Index:     i,
					Name:      tc.Function.Name,
					Arguments: parseArgumentsMap(json.RawMessage(jsonQuote(tc.Function.Arguments))),
				},
			})
		}
		toolName := m.Name
		if toolName == "" {
			toolName = m.ToolCallID
		}
		msgs = append(msgs, ChatMessage{
			Role:      m.Role,
			Content:   text,
			Images:    images,
			ToolCalls: tcs,
			ToolName:  toolName,
		})
	}

	var formatRaw json.RawMessage
	if rf := strings.TrimSpace(string(req.ResponseFormat)); rf != "" && rf != "null" {
		var parsed struct {
			Type       string `json:"type"`
			JSONSchema *struct {
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		}
		if err := json.Unmarshal(req.ResponseFormat, &parsed); err == nil {
			switch strings.ToLower(parsed.Type) {
			case "json_object":
				formatRaw = json.RawMessage(`"json"`)
			case "json_schema":
				if parsed.JSONSchema != nil && len(parsed.JSONSchema.Schema) > 0 {
					formatRaw = parsed.JSONSchema.Schema
				} else {
					formatRaw = json.RawMessage(`"json"`)
				}
			}
		}
	}

	opts := map[string]any{}
	if req.MaxCompletionTokens != nil {
		opts["num_predict"] = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		opts["num_predict"] = *req.MaxTokens
	}
	if len(strings.TrimSpace(string(req.Stop))) > 0 && string(req.Stop) != "null" {
		var stopAny any
		if json.Unmarshal(req.Stop, &stopAny) == nil {
			opts["stop"] = stopAny
		}
	}

	return ChatRequest{
		Model:    req.Model,
		Messages: msgs,
		Tools:    req.Tools,
		Format:   formatRaw,
		Options:  opts,
	}, nil
}

func decodeOpenAIContent(raw json.RawMessage) (string, []string, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return "", nil, nil
	}
	var str string
	if err := json.Unmarshal(raw, &str); err == nil {
		return str, nil, nil
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", nil, fmt.Errorf("unsupported message content format")
	}
	var textParts []string
	var images []string
	for _, p := range parts {
		switch p.Type {
		case "text", "input_text":
			if p.Text != "" {
				textParts = append(textParts, p.Text)
			}
		case "image_url", "input_image":
			if p.ImageURL != nil && strings.TrimSpace(p.ImageURL.URL) != "" {
				images = append(images, p.ImageURL.URL)
			}
		}
	}
	return strings.Join(textParts, "\n"), images, nil
}

func extractCompletionPrompt(raw json.RawMessage) string {
	var str string
	if err := json.Unmarshal(raw, &str); err == nil {
		return str
	}
	var arr []string
	if err := json.Unmarshal(raw, &arr); err == nil {
		return strings.Join(arr, "\n")
	}
	return ""
}

func jsonQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func prepareSSE(w http.ResponseWriter) (http.Flusher, bool) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "server_error", "streaming_unsupported", "streaming unsupported")
		return nil, false
	}
	w.WriteHeader(http.StatusOK)
	return flusher, true
}

func writeSSEData(w http.ResponseWriter, flusher http.Flusher, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func writeOpenAIModelError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, acp.ErrModelExcluded):
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", "model_excluded", err.Error())
	case errors.Is(err, acp.ErrModelNotFound):
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", "model_not_found", err.Error())
	default:
		writeOpenAIError(w, http.StatusInternalServerError, "server_error", "internal_error", err.Error())
	}
}

func writeOpenAIExecError(w http.ResponseWriter, err error) {
	if errors.Is(err, acp.ErrNativeToolBlocked) {
		writeOpenAIError(w, http.StatusForbidden, "policy_error", "native_tool_disabled", err.Error())
		return
	}
	writeOpenAIError(w, http.StatusBadGateway, "server_error", "upstream_acp_error", err.Error())
}

func writeOpenAIError(w http.ResponseWriter, status int, errType, code, msg string) {
	writeJSON(w, status, openAIErrorEnvelope{
		Error: openAIErrorDetail{
			Message: msg,
			Type:    errType,
			Code:    code,
		},
	})
}
