package extractor

type ExtractedFunc struct {
	Package    string
	Name       string
	Receiver   string   // empty if not a method
	Doc        string   // godoc comment
	Signature  string   // full signature line
	Body       string   // full function source
	Imports    []string // imports used by this function
	FilePath   string
	LineNumber int
}
