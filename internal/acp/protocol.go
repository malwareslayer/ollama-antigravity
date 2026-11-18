package acp

import "encoding/json"

// RPCMessage represents a generic JSON-RPC 2.0 frame (request, response, or notification).
type RPCMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// RPCError represents a JSON-RPC 2.0 error object.
type RPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	return e.Message
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

type initializeParams struct {
	ProtocolVersion    int            `json:"protocolVersion"`
	ClientCapabilities map[string]any `json:"clientCapabilities"`
	ClientInfo         clientInfo     `json:"clientInfo"`
}

type clientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type sessionNewParams struct {
	CWD        string          `json:"cwd"`
	MCPServers []any           `json:"mcpServers"`
	Meta       *sessionNewMeta `json:"_meta,omitempty"`
}

type sessionNewMeta struct {
	Antigravity *antigravityMeta `json:"antigravity,omitempty"`
}

type antigravityMeta struct {
	SystemPrompt *systemPromptMeta `json:"systemPrompt,omitempty"`
}

type systemPromptMeta struct {
	Append string `json:"append,omitempty"`
}

// AvailableModel represents a model entry returned by ACP session/new.
type AvailableModel struct {
	ModelID string `json:"modelId"`
	Name    string `json:"name"`
}

type sessionNewResult struct {
	SessionID string `json:"sessionId"`
	Models    struct {
		AvailableModels []AvailableModel `json:"availableModels"`
		CurrentModelID  string           `json:"currentModelId"`
	} `json:"models"`
}

type sessionSetModelParams struct {
	SessionID string `json:"sessionId"`
	ModelID   string `json:"modelId"`
}

// ContentBlock represents an ACP prompt content item (text or base64 image).
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

type sessionPromptParams struct {
	SessionID string         `json:"sessionId"`
	Prompt    []ContentBlock `json:"prompt"`
}

// Usage reports token accounting returned by session/prompt.
type Usage struct {
	InputTokens       int `json:"inputTokens"`
	OutputTokens      int `json:"outputTokens"`
	CachedReadTokens  int `json:"cachedReadTokens"`
	CachedWriteTokens int `json:"cachedWriteTokens"`
	ThoughtTokens     int `json:"thoughtTokens"`
	TotalTokens       int `json:"totalTokens"`
}

type sessionPromptResult struct {
	StopReason string `json:"stopReason"`
	Usage      Usage  `json:"usage"`
}

type sessionCloseParams struct {
	SessionID string `json:"sessionId"`
}

type sessionUpdateParams struct {
	SessionID string `json:"sessionId"`
	Update    struct {
		SessionUpdate string `json:"sessionUpdate"`
		ToolCallID    string `json:"toolCallId,omitempty"`
		Name          string `json:"name,omitempty"`
		Status        string `json:"status,omitempty"`
		Content       struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"update"`
}

type permissionRequestParams struct {
	Options []struct {
		OptionID string `json:"optionId"`
		Kind     string `json:"kind"`
	} `json:"options"`
}
