package parties.feedbackmediator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Feedback;
import genius.core.Vote;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiationWithAnOffer;
import genius.core.actions.GiveFeedback;
import genius.core.actions.NoAction;
import genius.core.actions.OfferForFeedback;
import genius.core.issue.Issue;
import genius.core.issue.Value;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.Mediator;
import genius.core.parties.NegotiationInfo;
import genius.core.protocol.DefaultMultilateralProtocol;
import genius.core.protocol.MediatorFeedbackBasedProtocol;
import parties.feedbackmediator.partialopponentmodel.PartialPreferenceModels;

public class SmartMediatorOnlyFeedback extends AbstractNegotiationParty implements Mediator {

	private Bid currentBid; // the current bid
	private Bid lastBid; // the bid that is offered in the previous round

	private Vote isAcceptable;
	private int lastAcceptedRoundNumber;
	private Bid lastAcceptedBid; // keeping the last accepted bid by all parties

	private int currentIndex;
	private List<GiveFeedback> currentFeedbackList;

	private int currentRound = 0;

	/*
	 * the model is learned by the mediator during the negotiation
	 */
	private PartialPreferenceModels preferenceList;
	private Random randomnr;

	public SmartMediatorOnlyFeedback() {

		super();
		lastAcceptedBid = null;
		currentBid = null;
		lastBid = null;
		currentFeedbackList = new ArrayList<GiveFeedback>();
		isAcceptable = Vote.ACCEPT;

	}

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);
		randomnr = new Random(info.getRandomSeed());
	}

	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {

		if (opponentAction instanceof GiveFeedback)
			currentFeedbackList.add((GiveFeedback) opponentAction);

	}

	@Override
	public Class<? extends DefaultMultilateralProtocol> getProtocol() {
		return MediatorFeedbackBasedProtocol.class;
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		currentRound++;
		/**
		 * FIXME this was calling negotiator.session.Timeline#isDeadlineReached
		 * which makes no sense, as we would not be here if the deadline had
		 * been reached. Instead we chose to use 5% time remaining
		 */
		if (timeline.getCurrentTime() > 0.95) {
			/*
			 * If the deadline is reached, end negotiation with last accepted
			 * bid by all parties..
			 */
			// writePreferenceStream.println("Session:"+getSessionNo());
			// writePreferenceStream.println(preferenceList.toString());
			// writePreferenceStream.println("**********************************************");

			System.out.println("Last Accepted Bid:" + lastAcceptedBid + " in " + lastAcceptedRoundNumber + "th round");

			return new EndNegotiationWithAnOffer(getPartyId(), lastAcceptedBid);
		}

		try {

			if (validActions.contains(NoAction.class)) {

				isAcceptable = Feedback.isAcceptable(currentFeedbackList());

				if (isAcceptable == Vote.ACCEPT) {
					lastAcceptedBid = new Bid(currentBid);
					lastAcceptedRoundNumber = currentRound;
				}

				return new NoAction(getPartyId());

			} else if (currentRound == 1) {
				/*
				 * initially generate a random bid and create the preference
				 * model
				 */
				currentBid = generateRandomBid();
				lastBid = new Bid(currentBid);
				preferenceList = new PartialPreferenceModels(new Bid(currentBid), null);// FIXME
				return (new OfferForFeedback(getPartyId(), currentBid));
				// when we stop searching part and start voting, inform the
				// agents about the last accepted bid by all
			} else {

				if (currentRound > 2) {

					preferenceList.updateIssuePreferenceList(currentIndex, lastBid.getValue(currentIndex),
							currentBid.getValue(currentIndex), currentFeedbackList);
					lastBid = new Bid(currentBid);
				}

				currentFeedbackList.clear();
				currentBid = modifyLastBid();

				if (currentBid == null) {
					/*
					 * if we are not able to generate bids;
					 */
					System.out.println("Last Accepted Round Number:" + lastAcceptedRoundNumber);
					System.out.println(preferenceList.toString());
					return (new EndNegotiationWithAnOffer(getPartyId(), lastAcceptedBid));
				}

			}
		} catch (Exception e) {
			System.out.println(
					currentRound + " :Cannnot generate random bid or update preference list problem:" + e.getMessage());

		}

		return (new OfferForFeedback(getPartyId(), currentBid));
	}

	/**
	 * modifies the last bid by changing the Nth issue value where N is the
	 * currentIndex. If not yet halfway the deadline, we use
	 * {@link #searchForNewValue()}, otherwise we pick randomly between
	 * {@link #searchForNewValue()} and {@link #getNashValue()}.
	 *
	 * @return the modified bid
	 */

	private Bid modifyLastBid() throws Exception {

		Bid modifiedBid = new Bid(lastBid);

		// epsilon= distance to deadline, in [1,0].
		double epsilon = 1f - getTimeLine().getTime();
		Value selectedValue = null;

		if (epsilon < 0.5 || (epsilon > Math.random())) {
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

	private Value searchForNewValue() throws Exception {

		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Issue currentIssue;
		Value newValue = null;

		ArrayList<Issue> checkedIssues = new ArrayList<Issue>(issues);
		int checkID;
		do {
			checkID = randomnr.nextInt(checkedIssues.size());
			currentIssue = checkedIssues.get(checkID);
			currentIndex = currentIssue.getNumber();
			newValue = preferenceList.getMissingValue(currentIndex);
			if (newValue != null)
				return newValue;
			checkedIssues.remove(checkID);
		} while (checkedIssues.size() > 0);

		checkedIssues = new ArrayList<Issue>(issues);
		do {
			checkID = randomnr.nextInt(checkedIssues.size());
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
			checkID = randomnr.nextInt(checkedIssues.size());
			currentIssue = checkedIssues.get(checkID);
			currentIndex = currentIssue.getNumber();

			ArrayList<Value> allValues = preferenceList.getAllPossibleValues(currentIndex);
			do {
				newValue = allValues.get(randomnr.nextInt(allValues.size()));
				if ((!newValue.equals(lastBid.getValue(currentIndex)))
						&& (preferenceList.mayImproveAll(currentIndex, lastBid.getValue(currentIndex), newValue)))
					return newValue;
				allValues.remove(newValue);
			} while (allValues.size() > 0);

			checkedIssues.remove(checkID);

		} while (checkedIssues.size() > 0);

		return null;
	}

	private Value getNashValue() throws Exception {
		List<Issue> issues = utilitySpace.getDomain().getIssues();
		Issue currentIssue;
		Value newValue = null;

		ArrayList<Issue> checkedIssues = new ArrayList<Issue>(issues);
		int checkID;

		do {
			checkID = randomnr.nextInt(checkedIssues.size());
			currentIssue = checkedIssues.get(checkID);
			currentIndex = currentIssue.getNumber();

			ArrayList<Value> allNashValues = preferenceList.getNashValues(currentIndex);
			do {
				newValue = allNashValues.get(randomnr.nextInt(allNashValues.size()));

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
	public String getDescription() {
		return "Smart Mediator Feedback only";
	}

}
