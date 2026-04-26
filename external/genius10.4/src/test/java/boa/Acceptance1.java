package boa;

import java.util.HashSet;
import java.util.Set;

import genius.core.boaframework.AcceptanceStrategy;
import genius.core.boaframework.Actions;
import genius.core.boaframework.BOAparameter;

public class Acceptance1 extends AcceptanceStrategy {

	@Override
	public Actions determineAcceptability() {
		return null;
	}

	@Override
	public String getName() {
		return "Accept 1";
	}

	public Set<BOAparameter> getParameterSpec() {
		Set<BOAparameter> params = new HashSet<BOAparameter>();

		params.add(new BOAparameter("p", 0.6, "acceptance1 param 1"));
		params.add(new BOAparameter("q", 0.3, "acceptance1 param 2"));
		return params;
	}

}
