package agents.nastyagent;

import java.util.List;
import java.util.Random;

import genius.core.actions.Action;
import genius.core.actions.Offer;

/**
 * Keeps repeating best bid as offer
 * 
 * @author W.Pasman
 *
 */
public class RandomBid extends NastyAgent {
	private Random random = new Random();

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		return new Offer(id, bids.get(random.nextInt(bids.size())));
	}
}
