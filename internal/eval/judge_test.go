package eval_test

import (
	"testing"

	"github.com/yavosh/gocoder/internal/eval"
)

func TestFormatJudgePrompt(t *testing.T) {
	prompt := "Write a Go function that wraps errors"
	response := "func wrap(err error) error { return fmt.Errorf(\"op: %w\", err) }"

	judgePrompt := eval.FormatJudgePrompt(prompt, response)
	if judgePrompt == "" {
		t.Error("expected non-empty judge prompt")
	}
}

func TestParseJudgeScore(t *testing.T) {
	response := `{"correctness": 4, "idiom": 5, "simplicity": 4, "completeness": 3}`
	score, err := eval.ParseJudgeScore(response)
	if err != nil {
		t.Fatalf("ParseJudgeScore: %v", err)
	}
	if score.Correctness != 4 {
		t.Errorf("expected correctness 4, got %d", score.Correctness)
	}
	if score.Average() < 3.0 || score.Average() > 5.0 {
		t.Errorf("unexpected average: %f", score.Average())
	}
}

func TestParseJudgeScoreInvalid(t *testing.T) {
	_, err := eval.ParseJudgeScore("not json")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}
