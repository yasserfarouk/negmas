package agents.anac.y2015.pnegotiator;

import java.util.Comparator;

import genius.core.issue.ValueDiscrete;

public class ValueFrequency {
	public ValueDiscrete value;
	public double utility;
	public int[] opponentFrequency;
	public int ourFrequency;

	public ValueFrequency(ValueDiscrete val, double util, int P) {
		value = val;
		utility = util;
		opponentFrequency = new int[P];
		for (int i = 0; i < opponentFrequency.length; i++)
			opponentFrequency[i] = 1;
		ourFrequency = 0;
	}
}

class VFComp implements Comparator<ValueFrequency> {
	@Override
	public int compare(ValueFrequency vf1, ValueFrequency vf2) {
		return (vf1.utility < vf2.utility) ? 1 : -1;
	}
}
/*
 * class VFComp implements Comparator<ValueDiscrete>{
 * 
 * @Override public int compare(ValueDiscrete v1, ValueDiscrete v2) { return
 * v1.value.compareTo(v2.value); } }
 */
