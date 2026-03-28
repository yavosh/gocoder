package pipeline_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/yavosh/gocoder/internal/pipeline"
)

func TestBuildFromDir(t *testing.T) {
	dir := t.TempDir()
	outputDir := t.TempDir()

	// Create a fake Go repo with real Go code
	goFile := `package greeting

import "fmt"

// Greet returns a greeting for the given name.
func Greet(name string) string {
	return fmt.Sprintf("Hello, %s!", name)
}

// Add returns the sum of two integers.
// It handles the common case of adding positive numbers.
func Add(a, b int) int {
	return a + b
}
`
	if err := os.WriteFile(filepath.Join(dir, "greeting.go"), []byte(goFile), 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := pipeline.BuildFromDir(context.Background(), dir, outputDir, pipeline.BuildOptions{
		MinLines: 2,
		FIMRatio: 0.5,
		Seed:     42,
	})
	if err != nil {
		t.Fatalf("BuildFromDir: %v", err)
	}

	if result.TotalExamples == 0 {
		t.Error("expected at least one training example")
	}
	if result.TrainExamples == 0 {
		t.Error("expected at least one train example")
	}

	// Check output files exist
	trainPath := filepath.Join(outputDir, "train.jsonl")
	if _, err := os.Stat(trainPath); err != nil {
		t.Errorf("expected train.jsonl to exist: %v", err)
	}
	evalPath := filepath.Join(outputDir, "eval.jsonl")
	if _, err := os.Stat(evalPath); err != nil {
		t.Errorf("expected eval.jsonl to exist: %v", err)
	}

	// Verify JSONL has content
	data, _ := os.ReadFile(trainPath)
	if len(data) == 0 {
		t.Error("train.jsonl is empty")
	}
}

func TestBuildFromDirSkipsVendor(t *testing.T) {
	dir := t.TempDir()
	outputDir := t.TempDir()

	// Create a Go file in vendor/ — should be skipped
	vendorDir := filepath.Join(dir, "vendor", "pkg")
	os.MkdirAll(vendorDir, 0o755)
	os.WriteFile(filepath.Join(vendorDir, "vendor.go"), []byte("package pkg\n\nfunc Vendor() {}\n"), 0o644)

	// Create a real Go file
	goFile := `package main

import "fmt"

// Hello prints a greeting to stdout.
// It uses fmt.Println for output.
func Hello() {
	fmt.Println("hello")
}
`
	os.WriteFile(filepath.Join(dir, "main.go"), []byte(goFile), 0o644)

	result, err := pipeline.BuildFromDir(context.Background(), dir, outputDir, pipeline.BuildOptions{
		MinLines: 2,
		FIMRatio: 0,
		Seed:     42,
	})
	if err != nil {
		t.Fatalf("BuildFromDir: %v", err)
	}

	// Should only have examples from main.go, not vendor/
	if result.TotalExamples != 1 {
		t.Errorf("expected 1 example (vendor skipped), got %d", result.TotalExamples)
	}
}
