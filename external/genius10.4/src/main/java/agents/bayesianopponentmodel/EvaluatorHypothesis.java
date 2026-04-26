package agents.bayesianopponentmodel;


import genius.core.utility.Evaluator;

public class EvaluatorHypothesis extends Hypothesis {
	private String fDesc;
	private Evaluator fEval;
	
	public EvaluatorHypothesis (Evaluator pEval) {
		fEval = pEval;
	}
	public Evaluator getEvaluator() {
		return fEval;
	}
	public String toString() {
		String lResult = "";
		lResult += fDesc;
		return lResult;
	}
	public void setDesc(String pValue) {
		fDesc = pValue;
	}
	
	public String getDesc() {
		return fDesc;
	}
}

