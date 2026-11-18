package config

import (
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds runtime settings loaded from environment variables.
type Config struct {
	ListenAddr           string
	AgyBin               string
	Workers              int
	Timeout              time.Duration
	EmptyCWD             string
	ExcludeModels        []string
	DefaultModel         string
	DisableTelemetry     bool
	IncludeDefaultCustom bool
	Sandbox              bool
	RestrictNativeTools  bool
	SessionReuse         bool
	SessionMaxTurns      int
	SessionMaxTokens     int
	LogLevel             slog.Level
}

// Load reads configuration from environment variables using safe defaults.
func Load() (Config, error) {
	cfg := Config{
		ListenAddr:           envOrDefault("LISTEN_ADDR", "0.0.0.0:11435"),
		AgyBin:               envOrDefault("AGY_BIN", "agy"),
		Workers:              2,
		Timeout:              120 * time.Second,
		EmptyCWD:             envOrDefault("ACP_EMPTY_CWD", "/tmp/hindsight-acp-empty"),
		ExcludeModels:        []string{"gemini-pro-orchestrator-g1-pro", "orchestrator"},
		DefaultModel:         envOrDefault("ACP_DEFAULT_MODEL", "claude-sonnet-5-5-medium"),
		DisableTelemetry:     true,
		IncludeDefaultCustom: false,
		Sandbox:              true,
		RestrictNativeTools:  true,
		SessionReuse:         true,
		SessionMaxTurns:      20,
		SessionMaxTokens:     120000,
		LogLevel:             slog.LevelInfo,
	}

	if v := strings.TrimSpace(os.Getenv("ACP_WORKERS")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return cfg, fmt.Errorf("invalid ACP_WORKERS=%q: must be positive integer", v)
		}
		cfg.Workers = n
	}

	if v := strings.TrimSpace(os.Getenv("ACP_TIMEOUT_SEC")); v != "" {
		sec, err := strconv.Atoi(v)
		if err != nil || sec < 1 {
			return cfg, fmt.Errorf("invalid ACP_TIMEOUT_SEC=%q: must be positive integer", v)
		}
		cfg.Timeout = time.Duration(sec) * time.Second
	}

	if raw, ok := os.LookupEnv("ACP_EXCLUDE_MODELS"); ok {
		cfg.ExcludeModels = parseCSV(raw)
	}

	if raw, ok := os.LookupEnv("ACP_DISABLE_TELEMETRY"); ok {
		b, err := strconv.ParseBool(strings.TrimSpace(raw))
		if err != nil {
			return cfg, fmt.Errorf("invalid ACP_DISABLE_TELEMETRY=%q: %w", raw, err)
		}
		cfg.DisableTelemetry = b
	}

	if raw, ok := os.LookupEnv("ACP_INCLUDE_DEFAULT_CUSTOMIZATIONS"); ok {
		b, err := strconv.ParseBool(strings.TrimSpace(raw))
		if err != nil {
			return cfg, fmt.Errorf("invalid ACP_INCLUDE_DEFAULT_CUSTOMIZATIONS=%q: %w", raw, err)
		}
		cfg.IncludeDefaultCustom = b
	}

	if raw, ok := os.LookupEnv("ACP_SANDBOX"); ok {
		b, err := strconv.ParseBool(strings.TrimSpace(raw))
		if err != nil {
			return cfg, fmt.Errorf("invalid ACP_SANDBOX=%q: %w", raw, err)
		}
		cfg.Sandbox = b
	}

	if raw, ok := os.LookupEnv("ACP_RESTRICT_NATIVE_TOOLS"); ok {
		b, err := strconv.ParseBool(strings.TrimSpace(raw))
		if err != nil {
			return cfg, fmt.Errorf("invalid ACP_RESTRICT_NATIVE_TOOLS=%q: %w", raw, err)
		}
		cfg.RestrictNativeTools = b
	}

	if raw, ok := os.LookupEnv("ACP_SESSION_REUSE"); ok {
		b, err := strconv.ParseBool(strings.TrimSpace(raw))
		if err != nil {
			return cfg, fmt.Errorf("invalid ACP_SESSION_REUSE=%q: %w", raw, err)
		}
		cfg.SessionReuse = b
	}

	if v := strings.TrimSpace(os.Getenv("ACP_SESSION_MAX_TURNS")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return cfg, fmt.Errorf("invalid ACP_SESSION_MAX_TURNS=%q: must be positive integer", v)
		}
		cfg.SessionMaxTurns = n
	}

	if v := strings.TrimSpace(os.Getenv("ACP_SESSION_MAX_TOKENS")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1000 {
			return cfg, fmt.Errorf("invalid ACP_SESSION_MAX_TOKENS=%q: must be >= 1000", v)
		}
		cfg.SessionMaxTokens = n
	}

	if raw := strings.TrimSpace(os.Getenv("LOG_LEVEL")); raw != "" {
		switch strings.ToLower(raw) {
		case "debug":
			cfg.LogLevel = slog.LevelDebug
		case "info":
			cfg.LogLevel = slog.LevelInfo
		case "warn", "warning":
			cfg.LogLevel = slog.LevelWarn
		case "error":
			cfg.LogLevel = slog.LevelError
		default:
			return cfg, fmt.Errorf("invalid LOG_LEVEL=%q: expected debug|info|warn|error", raw)
		}
	}

	return cfg, nil
}

func envOrDefault(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func parseCSV(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
