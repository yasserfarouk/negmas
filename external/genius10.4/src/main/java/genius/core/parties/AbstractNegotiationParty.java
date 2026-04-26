package genius.core.parties;

import java.util.HashMap;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Deadline;
import genius.core.Domain;
import genius.core.actions.Action;
import genius.core.actions.Inform;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.persistent.PersistentDataContainer;
import genius.core.protocol.MultilateralProtocol;
import genius.core.protocol.StackedAlternatingOffersProtocol;
import genius.core.timeline.TimeLineInfo;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.uncertainty.BidRanking;
import genius.core.uncertainty.User;
import genius.core.uncertainty.UserModel;
import genius.core.utility.AbstractUtilitySpace;

/**
 * A basic implementation of the {@link NegotiationParty} interface. This basic
 * implementation sets up some common variables for you.
 *
 * @author Tim Baarslag
 * @author David Festen
 * @author Reyhan (The random bid generator)
 */
public abstract class AbstractNegotiationParty implements NegotiationParty 
{
	/**
	 * Time line used by the party if time deadline is set.
	 */
	protected TimeLineInfo timeline;// should be final after init

	/**
	 * Random seed used by this party.
	 */
	protected Random rand;// should be final after init

	/**
	 * utility space used by this party (set in constructor).
	 * Used directly by lots of implementations.
	 */
	protected AbstractUtilitySpace utilitySpace;// should be final after init
	
	/** Instead of a {@link AbstractUtilitySpace}, the agent may receive a user model (i.e. uncertain preferences). */
	protected UserModel userModel;

	/**
	 * Under preference uncertainty, the agent will receive its corresponding user. 
	 */
	protected User user;
	/**
	 * Last received action, or null
	 */
	private Action lastReceivedAction = null;

	private int numberOfParties = -1;

	private NegotiationInfo info;

	@Override
	public void init(NegotiationInfo info) 
	{
		this.info = info;
		this.rand = new Random(info.getRandomSeed());
		this.timeline = info.getTimeline();
		this.userModel = info.getUserModel();
		this.user = info.getUser();
		
		// If the agent has uncertain preferences, the utility space provided to the agent by Genius will be null. 
		// In that case, the utility space is estimated with a simple heuristic so that any agent can
		// deal with preference uncertainty. This method can be overridden by the agent to provide better estimates.
		if (hasPreferenceUncertainty())
		{
			AbstractUtilitySpace passedUtilitySpace = info.getUtilitySpace();
			AbstractUtilitySpace estimatedUtilitySpace = estimateUtilitySpace();
			estimatedUtilitySpace.setReservationValue(passedUtilitySpace.getReservationValue());
			estimatedUtilitySpace.setDiscount(passedUtilitySpace.getDiscountFactor());
			info.setUtilSpace(estimatedUtilitySpace);
		}
		// Use either the provided utility space, or the hotswapped estimated utility space
		this.utilitySpace = info.getUtilitySpace();
	}

	/**
	 * Returns an estimate of the utility space given uncertain preferences specified by the user model.
	 * By default, the utility space is estimated with a simple counting heuristic so that any agent can 
	 * deal with preference uncertainty. 
	 *  
	 * This method can be overridden by the agent to provide better estimates. 
	 */
	public AbstractUtilitySpace estimateUtilitySpace()
	{
		return defaultUtilitySpaceEstimator(getDomain(), userModel);
	}
	
	/**
	 * Provides a simple estimate of a utility space given the partial preferences of a {@link UserModel}. 
	 * This is constructed as a static funtion so that other agents (that are not an {@link AbstractNegotiationParty})
	 * can also benfit from this functionality.
	 */
	public static AbstractUtilitySpace defaultUtilitySpaceEstimator(Domain domain, UserModel um)
	{
		AdditiveUtilitySpaceFactory factory = new AdditiveUtilitySpaceFactory(domain);
		BidRanking bidRanking = um.getBidRanking();
		factory.estimateUsingBidRanks(bidRanking);
		return factory.getUtilitySpace();
	}
	
	/**
	 * Returns the domain defined in either the utilityspace or user model of the agent. 
	 */
	public Domain getDomain()
	{
		if (utilitySpace != null)
			return utilitySpace.getDomain();
		return userModel.getDomain();
	}

	/**
	 * Generates a random bid which will be generated using this.utilitySpace.
	 *
	 * @return A random bid
	 */
	protected Bid generateRandomBid() {
		try {
			// Pairs <issue number, chosen value string>
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();

			// For each issue, put a random value
			for (Issue currentIssue : utilitySpace.getDomain().getIssues()) {
				values.put(currentIssue.getNumber(), getRandomValue(currentIssue));
			}

			// return the generated bid
			return new Bid(utilitySpace.getDomain(), values);

		} catch (Exception e) {

			// return empty bid if an error occurred
			return new Bid(utilitySpace.getDomain());
		}
	}

	/**
	 * Gets a random value for the given issue.
	 *
	 * @param currentIssue
	 *            The issue to generate a random value for
	 * @return The random value generated for the issue
	 * @throws Exception
	 *             if the issues type is not Discrete, Real or Integer.
	 */
	protected Value getRandomValue(Issue currentIssue) throws Exception {

		Value currentValue;
		int index;

		switch (currentIssue.getType()) {
		case DISCRETE:
			IssueDiscrete discreteIssue = (IssueDiscrete) currentIssue;
			index = (rand.nextInt(discreteIssue.getNumberOfValues()));
			currentValue = discreteIssue.getValue(index);
			break;
		case REAL:
			IssueReal realIss = (IssueReal) currentIssue;
			index = rand.nextInt(realIss.getNumberOfDiscretizationSteps()); // check
																			// this!
			currentValue = new ValueReal(
					realIss.getLowerBound() + (((realIss.getUpperBound() - realIss.getLowerBound()))
							/ (realIss.getNumberOfDiscretizationSteps())) * index);
			break;
		case INTEGER:
			IssueInteger integerIssue = (IssueInteger) currentIssue;
			index = rand.nextInt(integerIssue.getUpperBound() - integerIssue.getLowerBound() + 1);
			currentValue = new ValueInteger(integerIssue.getLowerBound() + index);
			break;
		default:
			throw new Exception("issue type " + currentIssue.getType() + " not supported");
		}

		return currentValue;
	}

	/**
	 * Gets the utility for the given bid.
	 *
	 * @param bid
	 *            The bid to get the utility for
	 * @return A double value between [0, 1] (inclusive) that represents the
	 *         bids utility
	 */
	public double getUtility(Bid bid) {
		try {
			// throws exception if bid incomplete or not in utility space
			return bid == null ? 0 : utilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		}
	}

	/**
	 * Gets the time discounted utility for the given bid.
	 *
	 * @param bid
	 *            The bid to get the utility for
	 * @return A double value between [0, 1] (inclusive) that represents the
	 *         bids utility
	 */
	public double getUtilityWithDiscount(Bid bid) {
		if (bid == null) {
			// utility is null if no bid
			return 0;
		} else if (timeline == null) {
			// return undiscounted utility if no timeline given
			return getUtility(bid);
		} else {
			// otherwise, return discounted utility
			return utilitySpace.getUtilityWithDiscount(bid, timeline);
		}
	}

	/**
	 * Gets this agent's utility space.
	 *
	 * @return The utility space
	 */
	public final AbstractUtilitySpace getUtilitySpace() {
		return utilitySpace;
	}

	/**
	 * Gets this agent's time line.
	 *
	 * @return The time line for this agent
	 */
	public TimeLineInfo getTimeLine() {
		return timeline;
	}

	/**
	 * Returns a human readable string representation of this party.
	 *
	 * @return the string representation of party id
	 */
	@Override
	public String toString() {
		return info.getAgentID().toString();
	}

	@Override
	public void receiveMessage(AgentID sender, Action act) {
		lastReceivedAction = act;
		if (act instanceof Inform) {
			numberOfParties = (Integer) ((Inform) act).getValue();
		}
	}

	/**
	 * 
	 * @return last received {@link Action} or null if nothing received yet.
	 */
	public Action getLastReceivedAction() {
		return lastReceivedAction;
	}

	public int getNumberOfParties() {
		if (numberOfParties == -1) {
			System.out.println("Make sure that you call the super class in receiveMessage() method.");
		}
		return numberOfParties;
	}

	final public AgentID getPartyId() {
		return info.getAgentID();
	}

	/**
	 * @return Whether the agent's preference profile has preference uncertainty enabled
	 */
	public boolean hasPreferenceUncertainty() 
	{
		return (userModel != null);		
	}
	
	@Override
	public Class<? extends MultilateralProtocol> getProtocol() {
		return StackedAlternatingOffersProtocol.class;
	}

	@Override
	public HashMap<String, String> negotiationEnded(Bid acceptedBid) {
		return null;
	}

	/**
	 * @return persistent data
	 */
	public PersistentDataContainer getData() {
		return info.getPersistentData();
	}

	public Deadline getDeadlines() {
		return info.getDeadline();
	}

	public UserModel getUserModel() {
		return userModel;
	}

}
