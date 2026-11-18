.PHONY: build test race vet fmt lint check

build:
	go build -o ollama-antigravity ./cmd/ollama-antigravity

test:
	go test -v ./...

race:
	go test -race ./...

vet:
	go vet ./...

fmt:
	gofmt -s -w .
	goimports -w .

lint:
	golangci-lint run ./...

check: vet lint race
