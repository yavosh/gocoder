package main

import (
	"os/exec"
	"strings"
	"testing"
)

func TestCLINoArgs(t *testing.T) {
	cmd := exec.Command("go", "run", ".")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("expected non-zero exit with no args, got nil error")
	}
	if cmd.ProcessState.ExitCode() == 0 {
		t.Fatalf("expected non-zero exit code, got 0. output: %s", out)
	}
}

func TestCLIUnknownCommand(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "foobar")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("expected non-zero exit with unknown command, got nil error")
	}
	if !strings.Contains(string(out), "unknown command") {
		t.Fatalf("expected output to contain 'unknown command', got: %s", out)
	}
}

func TestCLIPipelineBuild(t *testing.T) {
	outputDir := t.TempDir()
	cmd := exec.Command("go", "run", ".", "pipeline", "build", "--dir", ".", "--output", outputDir, "--min-lines", "2")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("expected pipeline build to succeed, got: %v\noutput: %s", err, out)
	}
	if !strings.Contains(string(out), "Pipeline build complete") {
		t.Fatalf("expected output to contain summary, got: %s", out)
	}
}
