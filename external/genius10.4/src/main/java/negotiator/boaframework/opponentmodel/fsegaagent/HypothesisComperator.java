package negotiator.boaframework.opponentmodel.fsegaagent;

import java.util.Comparator;
import agents.bayesianopponentmodel.Hypothesis;

public class HypothesisComperator  implements Comparator<Hypothesis> {
	
	public int compare(Hypothesis h1, Hypothesis h2) {
		if (h1.getProbability() > h2.getProbability())
			return 1;
		else if (h1.getProbability() < h2.getProbability())
	        return -1;
	    else
	        return 0;
	}
}

