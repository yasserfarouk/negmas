package parties.feedbackmediator;

import java.util.ArrayList;
import java.util.List;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.DeadlineType;
import genius.core.Feedback;
import genius.core.actions.Action;
import genius.core.actions.GiveFeedback;
import genius.core.actions.OfferForFeedback;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.Mediator;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.protocol.MediatorFeedbackBasedProtocol;
import genius.core.protocol.MultilateralProtocol;
import parties.feedbackmediator.partialopponentmodel.PartialPreferenceModels;

/**
 * Implementation of a mediator that uses feedback to make a (partial)
 * preference graph of the participating parties offers.
 * <p/>
 * This class was adapted from Reyhan's parties.SmartMediatorOnlyFeedback class
 * (see svn history for details about that class), and adapted to fit into the
 * new framework.
 *
 * @author W.Pasman (compatibility fixes)
 * @author David Festen
 * @author Reyhan (Orignal code)
 */
public class FeedbackMediator extends AbstractNegotiationParty implements Mediator {

	private List<GiveFeedback> currentFeedbackList;
	/**
	 * The current bid
	 */
	private Bid currentBid;
	/**
	 * The bid that is offered in the previous round
	 */
	private Bid lastBid;
	/**
	 * Keeping the last accepted bid by all parties
	 */
	private Bid lastAcceptedBid;
	/**
	 * The model is learned by the mediator during the negotiation
	 */
	private int roundNumber;

	/**
	 * The index number of the issue we are currently modifying.
	 */
	private int currentIndex;
	private PartialPreferenceModels preferenceList;

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		lastAcceptedBid = null;
		currentBid = null;
		lastBid = null;
		currentFeedbackList = new ArrayList<GiveFeedback>();
	}

	public Bid getLastAcceptedBid() {
		return lastAcceptedBid;
	}

	/**
	 * When this class is called, it is expected that the Party chooses one of
	 * the actions from the possible action list and returns an instance of the
	 * chosen action. This class is only called if this {@link NegotiationParty}
	 * is in the
	 * {@link genius.core.protocol.Protocol#getRoundStructure(java.util.List, negotiator.session.Session)}
	 * .
	 *
	 * @param possibleActions
	 *            List of all actions possible.
	 * @return The chosen action
	 */
	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		roundNumber++;

		if (currentBid == null) {
			/*
			 * initially generate a random bid and create the preference model
			 */
			currentBid = generateRandomBid();
			lastBid = new Bid(currentBid);
			return (new OfferForFeedback(getPartyId(), currentBid));
		} else {
			// we get here from round 2 onwards.
			if (roundNumber <= 2) {
				preferenceList = new PartialPreferenceModels(currentBid, currentFeedbackList);
			} else {
				/*
				 * Start updating the model from round 3 onward. The feedback on
				 * the first bid makes no sense, as agents need to have 2 bids
				 * before comparing them is possible.
				 */
				preferenceList.updateIssuePreferenceList(currentIndex, lastBid.getValue(currentIndex),
						currentBid.getValue(currentIndex), currentFeedbackList);
				lastBid = currentBid;
			}

			currentFeedbackList.clear();
			currentBid = modifyLastBid();

			if (currentBid == null) {
				throw new NullPointerException("internal error: currentBid=null!");
			}

		}

		return (new OfferForFeedback(getPartyId(), currentBid));
	}

	/**
	 * This method is called when an observable action is performed. Observable
	 * actions are defined in
	 * {@link MultilateralProtocol#getActionListeners(java.util.List)}
	 *
	 * @param sender
	 *            The initiator of the action
	 * @param action
	 *            The action performed by the sender
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		if (action instanceof GiveFeedback)
			currentFeedbackList.add((GiveFeedback) action);
	}

	private Bid modifyLastBid() {

		Bid modifiedBid = lastBid;

		// double epsilon = (double) (getTotalRoundOrTime() - getRound() - 1) /
		// getTotalRoundOrTime();
		double epsilon = getNormalizedRound();
		Value selectedValue = null;

		// if ((getRound() <= ((double) getTotalRoundOrTime() / 2)) || (epsilon
		// > Math.random()))
		if (epsilon > 0.5 || epsilon > rand.nextDouble()) {
			selectedValue = searchForNewValue();
		}

		if (selectedValue == null) {
			selectedValue = getNashValue();
		}

		if (selectedValue == null)
			return null;

		modifiedBid = modifiedBid.putValue(currentIndex, selectedValue);
		return modifiedBid;

	}

	/**
	 * @return round number between 0..1 or 1 if no round deadline set
	 */
	private double getNormalizedRound() {
		if (getDeadlines().getType() != DeadlineType.ROUND) {
			return 0d;
		}

		double totalRounds = getDeadlines().getValue();
		double currentRound = roundNumber;
		return (totalRounds - currentRound - 1d) / totalRounds;
	}

	private Value searchForNewValue() {

		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Issue currentIssue;
		Value newValue;

		ArrayList<Issue> checkedIssues = new ArrayList<Issue>(issues);
		int checkID;
		do {
			checkID = rand.nextInt(checkedIssues.size());
			currentIssue = checkedIssues.get(checkID);
			currentIndex = currentIssue.getNumber();
			newValue = preferenceList.getMissingValue(currentIndex);
			if (newValue != null)
				return newValue;
			checkedIssues.remove(checkID);
		} while (checkedIssues.size() > 0);

		checkedIssues = new ArrayList<Issue>(issues);
		do {
			checkID = rand.nextInt(checkedIssues.size());
			currentIssue = checkedIssues.get(checkID);
			currentIndex = currentIssue.getNumber();

			for (Value incomparable : preferenceList.getIncomparableValues(currentIndex,
					lastBid.getValue(currentIndex))) {
				if (incomparable != null)
					return incomparable;
			}
			checkedIssues.remove(checkID);

		} while (checkedIssues.size() > 0);

		checkedIssues = new ArrayList<Issue>(issues);
		do {
			checkID = rand.nextInt(checkedIssues.size());
			currentIssue = checkedIssues.get(checkID);
			currentIndex = currentIssue.getNumber();

			ArrayList<Value> allValues = preferenceList.getAllPossibleValues(currentIndex);
			do {
				newValue = allValues.get(rand.nextInt(allValues.size()));
				if ((!newValue.equals(lastBid.getValue(currentIndex)))
						&& (preferenceList.mayImproveAll(currentIndex, lastBid.getValue(currentIndex), newValue)))
					return newValue;
				allValues.remove(newValue);
			} while (allValues.size() > 0);

			checkedIssues.remove(checkID);

		} while (checkedIssues.size() > 0);

		return null;
	}

	private Value getNashValue() {
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Issue currentIssue;
		Value newValue;

		ArrayList<Issue> checkedIssues = new ArrayList<Issue>(issues);
		int checkID;

		do {
			checkID = rand.nextInt(checkedIssues.size());
			currentIssue = checkedIssues.get(checkID);
			currentIndex = currentIssue.getNumber();

			ArrayList<Value> allNashValues = preferenceList.getNashValues(currentIndex);
			do {
				newValue = allNashValues.get(rand.nextInt(allNashValues.size()));

				// if ((newValue!=lastBid.getValue(currentIndex)))
				if ((!newValue.equals(lastBid.getValue(currentIndex)))
						&& (preferenceList.mayImproveAll(currentIndex, lastBid.getValue(currentIndex), newValue)))
					return newValue;
				allNashValues.remove(newValue);
			} while (allNashValues.size() > 0);

			checkedIssues.remove(checkID);

		} while (checkedIssues.size() > 0);

		return newValue;

	}

	/**
	 * 
	 * @return just the Feedback elements of the currentFeedbacklist.
	 */
	private List<Feedback> currentFeedbackList() {
		List<Feedback> feedbacklist = new ArrayList<>();
		for (GiveFeedback feedback : currentFeedbackList) {
			feedbacklist.add(feedback.getFeedback());
		}
		return feedbacklist;
	}

	@Override
	public Class<? extends MultilateralProtocol> getProtocol() {
		return MediatorFeedbackBasedProtocol.class;
	}

	@Override
	public String getDescription() {
		return "Feedback Mediator";
	}
}
