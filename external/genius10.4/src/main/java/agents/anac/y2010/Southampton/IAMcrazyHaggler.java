package agents.anac.y2010.Southampton;

import java.util.Random;

import genius.core.Bid;
import genius.core.SupportedNegotiationSetting;

/**
 * @author Colin Williams
 * 
 */
public class IAMcrazyHaggler extends SouthamptonAgent {

	private double breakOff = 0.9;// 0.93;
	private final boolean TEST_EQUIVALENCE = false;
	private Random random100;

	@Override
	public void init() {
		super.init();
		MAXIMUM_ASPIRATION = 0.85;// 93;
		if (this.utilitySpace.isDiscounted()) {
			MAXIMUM_ASPIRATION = 0.9;
			breakOff = 0.95;
		}
		if (TEST_EQUIVALENCE) {
			random100 = new Random(100);
		} else {
			random100 = new Random();
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see negotiator.Agent#getName()
	 */
	@Override
	public String getName() {
		return "IAMcrazyHaggler";
	}

	@Override
	protected Bid proposeInitialBid() throws Exception {
		return proposeRandomBid();
	}

	@Override
	protected Bid proposeNextBid(Bid opponentBid) throws Exception {
		return proposeRandomBid();
	}

	private Bid proposeRandomBid() {
		Bid bid = null;
		try {
			do {
				bid = this.utilitySpace.getDomain().getRandomBid(random100);
			} while (this.utilitySpace.getUtility(bid) <= breakOff);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return bid;
	}

	/**
	 * @return
	 */
	@Override
	public String getVersion() {
		return "2.0 (Genius 3.1)";
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2010";
	}
}
