package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"ollama-antigravity/internal/acp"
	"ollama-antigravity/internal/config"
	"ollama-antigravity/internal/ollama"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal error", "err", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: cfg.LogLevel,
	}))
	slog.SetDefault(logger)

	logger.Info("starting ollama-antigravity bridge",
		"listen_addr", cfg.ListenAddr,
		"agy_bin", cfg.AgyBin,
		"workers", cfg.Workers,
		"timeout_sec", int(cfg.Timeout.Seconds()),
		"empty_cwd", cfg.EmptyCWD,
		"exclude_models", cfg.ExcludeModels,
		"default_model", cfg.DefaultModel,
	)

	rootCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	pool := acp.NewPool(cfg, logger)
	startCtx, cancelStart := context.WithTimeout(rootCtx, 45*time.Second)
	err = pool.Start(startCtx)
	cancelStart()
	if err != nil {
		return err
	}
	defer pool.Stop()

	srv := ollama.NewServer(pool, logger)
	httpSrv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 15 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		logger.Info("http server listening", "addr", cfg.ListenAddr)
		if listenErr := httpSrv.ListenAndServe(); listenErr != nil && !errors.Is(listenErr, http.ErrServerClosed) {
			errCh <- listenErr
		}
		close(errCh)
	}()

	select {
	case <-rootCtx.Done():
		logger.Info("shutdown signal received")
	case srvErr := <-errCh:
		if srvErr != nil {
			return srvErr
		}
	}

	shutCtx, cancelShut := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelShut()
	if err := httpSrv.Shutdown(shutCtx); err != nil {
		logger.Warn("http server shutdown error", "err", err)
	}
	pool.Stop()
	logger.Info("shutdown complete")
	return nil
}
