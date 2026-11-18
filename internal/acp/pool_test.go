package acp

import (
	"errors"
	"testing"
)

func sampleRawModels() []AvailableModel {
	return []AvailableModel{
		{ModelID: "gemini-4-argon-medium", Name: "Gemini 4 Argon (Medium)"},
		{ModelID: "gemini-pro-orchestrator-g1-pro", Name: "Orchestrator (Argon)"},
		{ModelID: "gemini-3.8-flash-low", Name: "Gemini 3.8 Flash (Low)"},
		{ModelID: "claude-opus-5-5-medium", Name: "Claude Opus 5.5 (Medium)"},
		{ModelID: "claude-sonnet-5-5-low", Name: "Claude Sonnet 5.5 (Low)"},
		{ModelID: "claude-sonnet-5-5-medium", Name: "Claude Sonnet 5.5 (Medium)"},
		{ModelID: "claude-sonnet-5-5-high", Name: "Claude Sonnet 5.5 (High)"},
		{ModelID: "gpt-oss-120b-medium", Name: "GPT-OSS 120B (Medium)"},
	}
}

func TestIsExcludedModel(t *testing.T) {
	excludes := []string{"gemini-pro-orchestrator-g1-pro", "orchestrator"}
	for _, m := range sampleRawModels() {
		got := IsExcludedModel(m, excludes)
		want := m.ModelID == "gemini-pro-orchestrator-g1-pro"
		if got != want {
			t.Errorf("IsExcludedModel(%q) = %v, want %v", m.ModelID, got, want)
		}
	}
}

func TestResolveFromCatalog(t *testing.T) {
	excludes := []string{"gemini-pro-orchestrator-g1-pro", "orchestrator"}
	raw := sampleRawModels()
	var catalog []AvailableModel
	for _, m := range raw {
		if !IsExcludedModel(m, excludes) {
			catalog = append(catalog, m)
		}
	}

	tests := []struct {
		name    string
		req     string
		wantID  string
		wantErr error
	}{
		{name: "exact claude", req: "claude-sonnet-5-5-low", wantID: "claude-sonnet-5-5-low"},
		{name: "latest suffix", req: "claude-sonnet-5-5-medium:latest", wantID: "claude-sonnet-5-5-medium"},
		{name: "base name fallback", req: "claude-sonnet-5-5", wantID: "claude-sonnet-5-5-medium"},
		{name: "short alias claude", req: "claude", wantID: "claude-sonnet-5-5-medium"},
		{name: "short alias opus", req: "claude-opus", wantID: "claude-opus-5-5-medium"},
		{name: "excluded orchestrator id", req: "gemini-pro-orchestrator-g1-pro", wantErr: ErrModelExcluded},
		{name: "excluded orchestrator latest", req: "gemini-pro-orchestrator-g1-pro:latest", wantErr: ErrModelExcluded},
		{name: "excluded orchestrator name", req: "Orchestrator (Argon)", wantErr: ErrModelExcluded},
		{name: "unknown model", req: "nonexistent-model", wantErr: ErrModelNotFound},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveFromCatalog(tc.req, catalog, raw, excludes)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("ResolveFromCatalog(%q) err = %v, want %v", tc.req, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveFromCatalog(%q) unexpected err = %v", tc.req, err)
			}
			if got.ModelID != tc.wantID {
				t.Fatalf("ResolveFromCatalog(%q) = %q, want %q", tc.req, got.ModelID, tc.wantID)
			}
		})
	}
}
