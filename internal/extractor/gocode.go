package extractor

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"strings"
)

// IsGeneratedFile checks if file-level header indicates generated code.
func IsGeneratedFile(src []byte) bool {
	first512 := src
	if len(first512) > 512 {
		first512 = first512[:512]
	}
	s := string(first512)
	return strings.Contains(s, "DO NOT EDIT") ||
		strings.Contains(s, "Code generated")
}

func ExtractFunctions(path string) ([]ExtractedFunc, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if IsGeneratedFile(src) {
		return nil, nil
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	var funcs []ExtractedFunc
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}

		ef := ExtractedFunc{
			Package:    f.Name.Name,
			Name:       fn.Name.Name,
			FilePath:   path,
			LineNumber: fset.Position(fn.Pos()).Line,
		}

		if fn.Recv != nil && len(fn.Recv.List) > 0 {
			var buf bytes.Buffer
			printer.Fprint(&buf, fset, fn.Recv.List[0].Type)
			ef.Receiver = buf.String()
		}

		if fn.Doc != nil {
			ef.Doc = strings.TrimSpace(fn.Doc.Text())
		}

		var bodyBuf bytes.Buffer
		printer.Fprint(&bodyBuf, fset, fn)
		ef.Body = bodyBuf.String()

		lines := strings.SplitN(ef.Body, "\n", 2)
		ef.Signature = strings.TrimSuffix(lines[0], " {")

		for _, imp := range f.Imports {
			ef.Imports = append(ef.Imports, strings.Trim(imp.Path.Value, `"`))
		}

		funcs = append(funcs, ef)
	}

	return funcs, nil
}
