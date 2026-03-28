package formatter

type TrainingExample struct {
	Text        string `json:"text,omitempty"`
	Instruction string `json:"instruction,omitempty"`
	Output      string `json:"output,omitempty"`
	Source      string `json:"source,omitempty"`
	Category    string `json:"category,omitempty"`
}
