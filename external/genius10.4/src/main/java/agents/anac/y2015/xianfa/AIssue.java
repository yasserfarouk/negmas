package agents.anac.y2015.xianfa;

import java.util.HashMap;
import java.util.Random;

import genius.core.issue.Value;

public class AIssue {
	
	HashMap<Value, Double> values = new HashMap<Value, Double>();
	int issnr;
	
	public AIssue(int n) {
		issnr = n;
	}
	
	public int getIssNr() {
		return issnr;
	}
	
	public void setVal(Value val, int total, int act) {
		double freq = values.get(val);
		if (act == 0) {
			values.put(val, ((freq*(total-1))+1)/total);
		} else {
			values.put(val, freq+0.13);
		}
		
	}

	public Value getDesiredVal() {
		double max = 0;
		Value value = new Value();
		for (Value v : values.keySet()) {
			if (values.get(v) > max) {
				max = values.get(v);
				value = v;
			}
		}
		return value;
	}
	
	public double getVal(Value val) {
		return values.get(val);
	}
	
	public HashMap<Value, Double> getValues() {
		return values;
	}
}
