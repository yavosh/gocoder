package eval

type ComparisonResult struct {
	ScoreDelta float64
	Improved   []string
	Regressed  []string
	Unchanged  []string
}

func Compare(baseline, candidate *RunOutput) ComparisonResult {
	baseScores := make(map[string]float64)
	for _, r := range baseline.Results {
		baseScores[r.PromptID] = r.Score
	}

	var result ComparisonResult
	result.ScoreDelta = candidate.Summary.AvgScore - baseline.Summary.AvgScore

	for _, r := range candidate.Results {
		base, ok := baseScores[r.PromptID]
		if !ok {
			continue
		}
		switch {
		case r.Score > base+0.05:
			result.Improved = append(result.Improved, r.PromptID)
		case r.Score < base-0.05:
			result.Regressed = append(result.Regressed, r.PromptID)
		default:
			result.Unchanged = append(result.Unchanged, r.PromptID)
		}
	}
	return result
}
