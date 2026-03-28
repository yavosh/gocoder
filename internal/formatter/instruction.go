package formatter

import (
	"fmt"

	"github.com/yavosh/gocoder/internal/extractor"
)

func FormatInstruction(f extractor.ExtractedFunc) TrainingExample {
	instruction := f.Doc
	if instruction == "" {
		instruction = fmt.Sprintf("Implement the Go function: %s", f.Signature)
	} else {
		instruction = fmt.Sprintf("%s\n\nSignature: %s", instruction, f.Signature)
	}

	return TrainingExample{
		Instruction: instruction,
		Output:      f.Body,
		Source:      f.FilePath,
		Category:    "instruction",
	}
}

func FormatProseInstruction(section extractor.ProseSection) []TrainingExample {
	var examples []TrainingExample

	for _, code := range section.CodeBlocks {
		examples = append(examples, TrainingExample{
			Instruction: section.Prose,
			Output:      code,
			Category:    "prose_to_code",
		})

		examples = append(examples, TrainingExample{
			Instruction: fmt.Sprintf("Explain this Go code:\n\n%s", code),
			Output:      section.Prose,
			Category:    "code_to_prose",
		})
	}

	return examples
}
