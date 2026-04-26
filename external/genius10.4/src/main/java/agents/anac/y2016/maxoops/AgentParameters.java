/*
 * Author: Max W. Y. Lam (Aug 1 2015)
 * Version: Milestone 1
 * 
 * */

package agents.anac.y2016.maxoops;

import java.util.HashMap;
import java.util.Random;

public class AgentParameters {

	Random rand = null;
	HashMap<String, Double> list = null;

	public AgentParameters() {
		this.rand = new Random();
		this.list = new HashMap<String, Double>();
	}

	public void addParam(String paramName, double paramVal) {
		this.list.put(paramName, paramVal);
	}

	public void addParam(String paramName, double upperBound, double lowerBound) {
		double paramVal = this.rand.nextDouble();
		paramVal = paramVal * (upperBound - lowerBound) + lowerBound;
		this.list.put(paramName, paramVal);
	}

	public double getParam(String paramName) {
		return this.list.get(paramName);
	}

}
