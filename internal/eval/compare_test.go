package eval_test

import (
	"testing"

	"github.com/yavosh/gocoder/internal/eval"
)

func TestCompare(t *testing.T) {
	baseline := &eval.RunOutput{
		Results: []eval.Result{
			{PromptID: "err_wrap", Score: 0.4},
			{PromptID: "err_sentinel", Score: 0.7},
		},
		Summary: eval.Summary{AvgScore: 0.55},
	}

	candidate := &eval.RunOutput{
		Results: []eval.Result{
			{PromptID: "err_wrap", Score: 0.7},
			{PromptID: "err_sentinel", Score: 0.7},
		},
		Summary: eval.Summary{AvgScore: 0.70},
	}

	diff := eval.Compare(baseline, candidate)
	if diff.ScoreDelta <= 0 {
		t.Errorf("expected positive score delta, got %f", diff.ScoreDelta)
	}
	if len(diff.Improved) != 1 {
		t.Errorf("expected 1 improved prompt, got %d", len(diff.Improved))
	}
	if len(diff.Unchanged) != 1 {
		t.Errorf("expected 1 unchanged prompt, got %d", len(diff.Unchanged))
	}
	if len(diff.Regressed) != 0 {
		t.Errorf("expected 0 regressed prompts, got %d", len(diff.Regressed))
	}
}
