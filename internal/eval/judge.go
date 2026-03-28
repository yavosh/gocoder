package eval

import (
	"context"
	"encoding/json"
	"fmt"
)

type JudgeScore struct {
	Correctness  int `json:"correctness"`
	Idiom        int `json:"idiom"`
	Simplicity   int `json:"simplicity"`
	Completeness int `json:"completeness"`
}

func (s JudgeScore) Average() float64 {
	return float64(s.Correctness+s.Idiom+s.Simplicity+s.Completeness) / 4.0
}

func FormatJudgePrompt(originalPrompt, modelResponse string) string {
	return fmt.Sprintf(`Rate this Go code on a scale of 1-5 for each criterion.
Respond with JSON only: {"correctness": N, "idiom": N, "simplicity": N, "completeness": N}

Original prompt: %s

Code to evaluate:
%s`, originalPrompt, modelResponse)
}

func ParseJudgeScore(response string) (JudgeScore, error) {
	var score JudgeScore
	if err := json.Unmarshal([]byte(response), &score); err != nil {
		return score, fmt.Errorf("parsing judge score: %w", err)
	}
	return score, nil
}

func JudgeResult(ctx context.Context, endpoint, judgeModel, prompt, response string) (JudgeScore, error) {
	judgePrompt := FormatJudgePrompt(prompt, response)
	judgeResponse, err := RunPrompt(ctx, endpoint, judgeModel, judgePrompt)
	if err != nil {
		return JudgeScore{}, fmt.Errorf("running judge: %w", err)
	}
	return ParseJudgeScore(judgeResponse)
}
