.PHONY: build test lint clean

build:
	go build -o bin/gocoder ./cmd/gocoder

test:
	go test ./...

lint:
	go vet ./...
	staticcheck ./...

clean:
	rm -rf bin/
