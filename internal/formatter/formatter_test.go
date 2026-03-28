package formatter_test

import (
	"strings"
	"testing"

	"github.com/yavosh/gocoder/internal/extractor"
	"github.com/yavosh/gocoder/internal/formatter"
)

func TestFormatFIM(t *testing.T) {
	f := extractor.ExtractedFunc{
		Body: "func Add(a, b int) int {\n\treturn a + b\n}",
	}

	example, err := formatter.FormatFIM(f, 42)
	if err != nil {
		t.Fatalf("FormatFIM: %v", err)
	}

	if example.Text == "" {
		t.Fatal("expected non-empty FIM text")
	}

	for _, token := range []string{"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"} {
		if !strings.Contains(example.Text, token) {
			t.Errorf("expected FIM text to contain %s", token)
		}
	}

	if example.Category != "fim" {
		t.Errorf("expected category fim, got %q", example.Category)
	}
}

func TestFormatFIMDeterministic(t *testing.T) {
	f := extractor.ExtractedFunc{
		Body: "func Add(a, b int) int {\n\treturn a + b\n}",
	}

	ex1, _ := formatter.FormatFIM(f, 42)
	ex2, _ := formatter.FormatFIM(f, 42)

	if ex1.Text != ex2.Text {
		t.Error("expected deterministic output for same seed")
	}
}

func TestFormatInstruction(t *testing.T) {
	f := extractor.ExtractedFunc{
		Doc:       "Add returns the sum of a and b.",
		Signature: "func Add(a, b int) int",
		Body:      "func Add(a, b int) int {\n\treturn a + b\n}",
	}

	example := formatter.FormatInstruction(f)

	if example.Instruction == "" {
		t.Error("expected non-empty instruction")
	}
	if example.Output == "" {
		t.Error("expected non-empty output")
	}
	if !strings.Contains(example.Instruction, "Add returns") {
		t.Error("instruction should contain doc comment")
	}
	if example.Category != "instruction" {
		t.Errorf("expected category instruction, got %q", example.Category)
	}
}

func TestFormatInstructionNoDoc(t *testing.T) {
	f := extractor.ExtractedFunc{
		Signature: "func Add(a, b int) int",
		Body:      "func Add(a, b int) int {\n\treturn a + b\n}",
	}

	example := formatter.FormatInstruction(f)
	if !strings.Contains(example.Instruction, "Implement") {
		t.Error("expected fallback instruction with 'Implement'")
	}
}

func TestFormatProseInstruction(t *testing.T) {
	section := extractor.ProseSection{
		Prose:      "In Go, errors are values. Always wrap with context.",
		CodeBlocks: []string{"if err != nil {\n\treturn fmt.Errorf(\"op: %w\", err)\n}"},
	}

	examples := formatter.FormatProseInstruction(section)
	if len(examples) == 0 {
		t.Fatal("expected at least one example from prose section")
	}
	if len(examples) != 2 {
		t.Errorf("expected 2 examples (prose->code + code->prose), got %d", len(examples))
	}
}
