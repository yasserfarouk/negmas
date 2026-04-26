package agents.anac.y2012.AgentLG;
import java.util.Comparator;

import genius.core.Bid;
import genius.core.utility.AdditiveUtilitySpace;


public class BidsComparator  implements Comparator<Bid> {
	
	public BidsComparator(AdditiveUtilitySpace utilitySpace) {
		super();
		this.utilitySpace = utilitySpace;
	}

	private AdditiveUtilitySpace  utilitySpace;
	
	@Override
	public int compare(Bid arg0, Bid arg1) {
		try {
			if (utilitySpace.getUtility(arg0) < utilitySpace.getUtility(arg1))
					return 1;
			else if (utilitySpace.getUtility(arg0) == (utilitySpace.getUtility(arg1))) {
				return 0;
			}
		} catch (Exception e) {

		}
		return -1;
	}

}
