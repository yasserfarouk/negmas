package negotiator.parties;

import java.util.List;
import java.util.Random;

import genius.core.Bid;
import genius.core.bidding.BidDetails;
import genius.core.misc.Range;
import genius.core.parties.NegotiationInfo;

public class NonDeterministicConcederNegotiationParty extends AbstractTimeDependentNegotiationParty {

	public static final double DELTA = 0.05;
	protected Random random;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		random = new Random();
	}

	@Override
	protected Bid getNextBid() {
		final List<BidDetails> candidates = getCandidates(getTargetUtility(), DELTA);
		final BidDetails chosen = getRandomElement(candidates);
		return chosen.getBid();
	}

	protected List<BidDetails> getCandidates(double target, double delta) {
		return outcomeSpace.getBidsinRange(new Range(target - delta, target + delta));
	}

	protected <T> T getRandomElement(List<T> list) {
		return list.get(random.nextInt(list.size()));
	}

	@Override
	public double getE() {
		return 2;
	}

	@Override
	public String getDescription() {
		return "Nondeterministic Conceder Party";
	}
}
