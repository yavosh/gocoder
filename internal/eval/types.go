package eval

type Prompt struct {
	ID          string  `yaml:"id"`
	Category    string  `yaml:"category"`
	Description string  `yaml:"description"`
	Prompt      string  `yaml:"prompt"`
	Weight      float64 `yaml:"weight"`
}

type PromptFile struct {
	Category string   `yaml:"category"`
	Weight   float64  `yaml:"weight"`
	Prompts  []Prompt `yaml:"prompts"`
}

type Result struct {
	PromptID  string  `json:"prompt_id"`
	Category  string  `json:"category"`
	Response  string  `json:"response"`
	Compiles  bool    `json:"compiles"`
	VetClean  bool    `json:"vet_clean"`
	StaticOK  bool    `json:"static_ok"`
	TestsPass bool    `json:"tests_pass"`
	Score     float64 `json:"score"`
}

type RunOutput struct {
	Model   string   `json:"model"`
	Results []Result `json:"results"`
	Summary Summary  `json:"summary"`
}

type Summary struct {
	TotalPrompts int     `json:"total_prompts"`
	AvgScore     float64 `json:"avg_score"`
	CompileRate  float64 `json:"compile_rate"`
	VetRate      float64 `json:"vet_rate"`
}
