package agents.anac.y2016.caduceus.agents.Caduceus;

import java.util.ArrayList;

import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneBid;
import agents.anac.y2016.caduceus.agents.Caduceus.sanity.SaneUtilitySpace;
import genius.core.Bid;
import genius.core.BidIterator;
import genius.core.utility.UtilitySpace;

/**
 * Created by burakatalay on 20/03/16.
 */
public class NashProductCalculator {
	ArrayList<SaneUtilitySpace> utilitySpaces;

	double nashProduct = 0.0;
	Bid nashBid = null;

	public NashProductCalculator(ArrayList<SaneUtilitySpace> utilitySpaces)
			throws Exception {
		this.utilitySpaces = utilitySpaces;

		for (SaneUtilitySpace utilitySpace : utilitySpaces) {
			utilitySpace.normalize();
		}
	}

	public void calculate(UtilitySpace space) throws Exception {
		double tempProduct = 1.0D;

		BidIterator bidIterator = new BidIterator(space.getDomain());
		int i = 0;
		int count = 0;

		while (bidIterator.hasNext()) {
			++i;

			Bid currentBid = bidIterator.next();
			SaneBid saneBid = new SaneBid(currentBid, utilitySpaces.get(0));

			for (int index = 0; index < this.utilitySpaces.size(); ++index) {
				double u = (this.utilitySpaces.get(index))
						.getBidUtility(saneBid);
				tempProduct *= u;
			}

			if (tempProduct == 1.0D) {
				++count;
			}

			if (tempProduct > nashProduct) {
				nashProduct = tempProduct;
				nashBid = currentBid;
			}
		}
	}
}
