package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ollama-antigravity/internal/acp"
)

type mockExecutor struct {
	raw      []acp.AvailableModel
	excludes []string
	eventsFn func(req acp.PromptRequest) []acp.StreamEvent
	lastReq  acp.PromptRequest
}

func (m *mockExecutor) ListModels() []acp.AvailableModel {
	out := make([]acp.AvailableModel, 0, len(m.raw))
	for _, mod := range m.raw {
		if !acp.IsExcludedModel(mod, m.excludes) {
			out = append(out, mod)
		}
	}
	return out
}

func (m *mockExecutor) ResolveModel(requested string) (acp.AvailableModel, error) {
	return acp.ResolveFromCatalog(requested, m.ListModels(), m.raw, m.excludes)
}

func (m *mockExecutor) Execute(_ context.Context, req acp.PromptRequest) (<-chan acp.StreamEvent, error) {
	m.lastReq = req
	ch := make(chan acp.StreamEvent, 16)
	go func() {
		defer close(ch)
		for _, ev := range m.eventsFn(req) {
			ch <- ev
		}
	}()
	return ch, nil
}

func newTestServer(fn func(acp.PromptRequest) []acp.StreamEvent) (*Server, *mockExecutor) {
	exec := &mockExecutor{
		raw: []acp.AvailableModel{
			{ModelID: "claude-sonnet-5-5-medium", Name: "Claude Sonnet 5.5 (Medium)"},
			{ModelID: "gemini-pro-orchestrator-g1-pro", Name: "Orchestrator (Argon)"},
		},
		excludes: []string{"gemini-pro-orchestrator-g1-pro", "orchestrator"},
		eventsFn: fn,
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return NewServer(exec, logger), exec
}

func TestTagsAndShowExcludeOrchestrator(t *testing.T) {
	srv, _ := newTestServer(nil)
	h := srv.Handler()

	// GET /api/tags
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/tags status = %d", rec.Code)
	}
	var tags TagsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &tags); err != nil {
		t.Fatalf("decode tags: %v", err)
	}
	if len(tags.Models) != 1 || tags.Models[0].Model != "claude-sonnet-5-5-medium" {
		t.Fatalf("unexpected tags models: %+v", tags.Models)
	}

	// POST /api/show for excluded model
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"gemini-pro-orchestrator-g1-pro"}`)))
	if rec2.Code != http.StatusNotFound {
		t.Fatalf("POST /api/show excluded status = %d, want 404", rec2.Code)
	}

	// POST /api/show for valid Claude model
	rec3 := httptest.NewRecorder()
	h.ServeHTTP(rec3, httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"claude-sonnet-5-5-medium:latest"}`)))
	if rec3.Code != http.StatusOK {
		t.Fatalf("POST /api/show valid status = %d", rec3.Code)
	}
	var show ShowResponse
	if err := json.Unmarshal(rec3.Body.Bytes(), &show); err != nil {
		t.Fatalf("decode show: %v", err)
	}
	if show.Details.Family != "claude" {
		t.Fatalf("show family = %q, want claude", show.Details.Family)
	}
}

func TestChatNonStreamingAndJSONFenceStrip(t *testing.T) {
	srv, exec := newTestServer(func(_ acp.PromptRequest) []acp.StreamEvent {
		return []acp.StreamEvent{
			{Type: acp.EventThoughtChunk, Text: "reasoning"},
			{Type: acp.EventTextChunk, Text: "```json\n"},
			{Type: acp.EventTextChunk, Text: `{"ok":true}`},
			{Type: acp.EventTextChunk, Text: "\n```"},
			{Type: acp.EventDone, Usage: acp.Usage{InputTokens: 10, OutputTokens: 5}, LoadDur: 10 * time.Millisecond, EvalDur: 20 * time.Millisecond},
		}
	})

	body := `{
		"model": "claude-sonnet-5-5-medium",
		"stream": false,
		"think": true,
		"format": "json",
		"messages": [
			{"role": "system", "content": "Extract JSON."},
			{"role": "user", "content": "Go"}
		]
	}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /api/chat status = %d, body=%s", rec.Code, rec.Body.String())
	}

	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode chat response: %v", err)
	}
	if resp.Message.Content != `{"ok":true}` {
		t.Fatalf("content = %q, want {\"ok\":true}", resp.Message.Content)
	}
	if resp.Message.Thinking != "reasoning" {
		t.Fatalf("thinking = %q, want reasoning", resp.Message.Thinking)
	}
	if !resp.Done || resp.EvalCount != 5 || resp.PromptEvalCount != 10 {
		t.Fatalf("unexpected metadata: %+v", resp)
	}
	if !strings.Contains(exec.lastReq.CallerSystem, "Extract JSON.") {
		t.Fatalf("caller system prompt missing user system message: %q", exec.lastReq.CallerSystem)
	}
}

func TestChatStreamingNDJSON(t *testing.T) {
	srv, _ := newTestServer(func(_ acp.PromptRequest) []acp.StreamEvent {
		return []acp.StreamEvent{
			{Type: acp.EventTextChunk, Text: "Hello "},
			{Type: acp.EventTextChunk, Text: "world"},
			{Type: acp.EventDone, Usage: acp.Usage{InputTokens: 4, OutputTokens: 2}},
		}
	})

	body := `{"model":"claude-sonnet-5-5-medium","messages":[{"role":"user","content":"Hi"}]}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.Contains(ct, "application/x-ndjson") {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", ct)
	}

	scanner := bufio.NewScanner(bytes.NewReader(rec.Body.Bytes()))
	var lines []ChatResponse
	for scanner.Scan() {
		var r ChatResponse
		if err := json.Unmarshal(scanner.Bytes(), &r); err != nil {
			t.Fatalf("decode ndjson line: %v", err)
		}
		lines = append(lines, r)
	}
	if len(lines) != 3 || lines[0].Message.Content != "Hello " || lines[1].Message.Content != "world" || !lines[2].Done {
		t.Fatalf("unexpected ndjson lines: %+v", lines)
	}
}

func TestChatToolCalls(t *testing.T) {
	srv, _ := newTestServer(func(_ acp.PromptRequest) []acp.StreamEvent {
		return []acp.StreamEvent{
			{Type: acp.EventTextChunk, Text: `{"tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Jakarta"}}}]}`},
			{Type: acp.EventDone},
		}
	})

	body := `{
		"model": "claude-sonnet-5-5-medium",
		"stream": false,
		"tools": [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],
		"messages": [{"role":"user","content":"Weather in Jakarta?"}]
	}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Message.Content != "" || len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call and empty content, got: %+v", resp.Message)
	}
	if resp.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("tool name = %q", resp.Message.ToolCalls[0].Function.Name)
	}
}

func TestChatHermesToolCallTagsAndEmbeddedCloseTag(t *testing.T) {
	srv, _ := newTestServer(func(_ acp.PromptRequest) []acp.StreamEvent {
		return []acp.StreamEvent{
			{Type: acp.EventTextChunk, Text: `<tool_call>{"id":"call_1","type":"function","function":{"name":"write_note","arguments":"{\"body\":\"literal </tool_call> inside string\"}"}}</tool_call>`},
			{Type: acp.EventDone},
		}
	})

	body := `{
		"model": "claude-sonnet-5-5-medium",
		"stream": false,
		"tools": [{"type":"function","function":{"name":"write_note","parameters":{"type":"object"}}}],
		"messages": [{"role":"user","content":"Save note"}]
	}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %+v", resp.Message)
	}
	if got := resp.Message.ToolCalls[0].Function.Arguments["body"]; got != "literal </tool_call> inside string" {
		t.Fatalf("unexpected decoded arg body = %v", got)
	}
}

func TestChatNativeToolBlockedReturns403(t *testing.T) {
	srv, _ := newTestServer(func(_ acp.PromptRequest) []acp.StreamEvent {
		return []acp.StreamEvent{
			{Type: acp.EventError, Err: fmt.Errorf("%w: run_command", acp.ErrNativeToolBlocked)},
		}
	})

	body := `{"model":"claude-sonnet-5-5-medium","stream":false,"messages":[{"role":"user","content":"Run uname"}]}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403, body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "run_command") && !errors.Is(acp.ErrNativeToolBlocked, acp.ErrNativeToolBlocked) {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}
}

func TestV1ModelsAndChatCompletions(t *testing.T) {
	srv, _ := newTestServer(func(_ acp.PromptRequest) []acp.StreamEvent {
		return []acp.StreamEvent{
			{Type: acp.EventThoughtChunk, Text: "step1"},
			{Type: acp.EventTextChunk, Text: `{"answer":42}`},
			{Type: acp.EventDone, Usage: acp.Usage{InputTokens: 12, OutputTokens: 6}},
		}
	})
	h := srv.Handler()

	// GET /v1/ and GET /v1/models should both return the filtered list
	for _, path := range []string{"/v1", "/v1/", "/v1/models"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d", path, rec.Code)
		}
		var list openAIModelsList
		if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
			t.Fatalf("decode %s: %v", path, err)
		}
		if list.Object != "list" || len(list.Data) != 1 || list.Data[0].ID != "claude-sonnet-5-5-medium" {
			t.Fatalf("unexpected models list at %s: %+v", path, list)
		}
	}

	// GET /v1/models/{excluded} -> 404
	recEx := httptest.NewRecorder()
	h.ServeHTTP(recEx, httptest.NewRequest(http.MethodGet, "/v1/models/gemini-pro-orchestrator-g1-pro", nil))
	if recEx.Code != http.StatusNotFound {
		t.Fatalf("GET /v1/models/excluded status = %d, want 404", recEx.Code)
	}

	// POST /v1/chat/completions non-streaming
	reqBody := `{
		"model": "claude-sonnet-5-5-medium",
		"response_format": {"type": "json_object"},
		"messages": [{"role": "user", "content": "Answer"}]
	}`
	recChat := httptest.NewRecorder()
	h.ServeHTTP(recChat, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(reqBody)))
	if recChat.Code != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions status = %d, body=%s", recChat.Code, recChat.Body.String())
	}
	var comp openAIChatResponse
	if err := json.Unmarshal(recChat.Body.Bytes(), &comp); err != nil {
		t.Fatalf("decode v1 chat: %v", err)
	}
	if comp.Object != "chat.completion" || len(comp.Choices) != 1 || comp.Choices[0].Message.Content != `{"answer":42}` {
		t.Fatalf("unexpected v1 chat response: %+v", comp)
	}
	if comp.Usage == nil || comp.Usage.TotalTokens != 18 {
		t.Fatalf("unexpected usage: %+v", comp.Usage)
	}

	// POST /v1/chat/completions streaming SSE
	streamBody := `{"model":"claude-sonnet-5-5-medium","stream":true,"messages":[{"role":"user","content":"Hi"}]}`
	recSSE := httptest.NewRecorder()
	h.ServeHTTP(recSSE, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(streamBody)))
	if recSSE.Code != http.StatusOK {
		t.Fatalf("POST /v1/chat/completions stream status = %d", recSSE.Code)
	}
	if ct := recSSE.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("Content-Type = %q", ct)
	}
	if !strings.Contains(recSSE.Body.String(), "data: [DONE]") {
		t.Fatalf("missing SSE [DONE] marker: %s", recSSE.Body.String())
	}
}
