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

func TestCLIPipelineStub(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "pipeline", "build")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("expected non-zero exit for stub, got nil error")
	}
	if !strings.Contains(string(out), "not implemented") {
		t.Fatalf("expected output to contain 'not implemented', got: %s", out)
	}
}
