package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

type RunConfig struct {
	Endpoint  string
	Model     string
	OutputDir string
}

func LoadPrompts(promptsDir string) ([]Prompt, error) {
	files, err := filepath.Glob(filepath.Join(promptsDir, "*.yaml"))
	if err != nil {
		return nil, err
	}

	var all []Prompt
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", f, err)
		}

		var pf PromptFile
		if err := yaml.Unmarshal(data, &pf); err != nil {
			return nil, fmt.Errorf("parsing %s: %w", f, err)
		}

		for i := range pf.Prompts {
			pf.Prompts[i].Category = pf.Category
			if pf.Prompts[i].Weight == 0 {
				pf.Prompts[i].Weight = pf.Weight
			}
		}
		all = append(all, pf.Prompts...)
	}
	return all, nil
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

func RunPrompt(ctx context.Context, endpoint, model, prompt string) (string, error) {
	reqBody := chatRequest{
		Model: model,
		Messages: []chatMessage{
			{Role: "system", Content: "You are an expert Go developer. Respond with Go code only, no explanations."},
			{Role: "user", Content: prompt},
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	url := endpoint + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, body)
	}

	var chatResp chatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("parsing response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, nil
}
