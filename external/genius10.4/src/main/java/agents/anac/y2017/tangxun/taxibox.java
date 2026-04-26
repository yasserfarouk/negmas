package agents.anac.y2017.tangxun;

import java.util.List;

import java.util.ArrayList;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Offer;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;

/**
 * This is your negotiation party.
 */
public class taxibox extends AbstractNegotiationParty {
	private double emax, target, shiwang, taxi, target1, target2, A, wangu1, B,
			avg, P, Z, Y, M, C, zishen, yingxiang, target3, x, wangu;
	private Bid lastReceivedBid = null;

	@Override
	public void init(NegotiationInfo info) {
		double t = info.getTimeline().getTime();
		super.init(info);

		List<Double> arr = new ArrayList<Double>();// 收集对手提案在我方的效用值
		for (int i = 0; i <= 1000; i++) {
			if (i < 1) {
				B = 0.1;
			} else

				arr.add(B);
			if (i > 0) {
				P = arr.get(i - 1);
			}
			System.out.println(B);
		}
		double sum = 0;

		for (int i = 0; i < arr.size(); i++) {
			sum = sum + arr.get(i);

			avg = sum / arr.size();// 对手提案在我方的效用值的平均值
			if (i > 0) {
				A = avg;
			}

		}
		wangu = B - P;
		wangu1 = Math.abs(B - P);

		emax = A + (1 - A) * wangu1;
		target = 1 - (1 - emax) * Math.pow(t, 3);
		// 例：時間依存の線形な譲歩関数

		shiwang = target - B;
		target1 = 1 - (1 - emax) * Math.pow(t, (2.7 + shiwang - t));

		System.out.println(target1);

		getUtilitySpace().getDomain().getIssues().size();

		// if you need to initialize some variables, please initialize them
		// below

	}

	/**
	 * Each round this method gets called and ask you to accept or offer. The
	 * first party in the first round is a bit different, it can only propose an
	 * offer.
	 *
	 * @param validActions
	 *            Either a list containing both accept and offer or only offer.
	 * @return The chosen action.
	 */

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		double t = timeline.getTime();
		List<Double> arr = new ArrayList<Double>();// 收集对手提案在我方的效用值
		for (int i = 0; i <= 1000; i++) {
			if (i < 1) {
				B = 0.1;
			} else

				arr.add(B);
			if (i > 0) {
				P = arr.get(i - 1);
			}
			System.out.println(B);
		}
		double sum = 0;

		for (int i = 0; i < arr.size(); i++) {
			sum = sum + arr.get(i);

			avg = sum / arr.size();// 对手提案在我方的效用值的平均值
			if (i > 0) {
				A = avg;
			}

		}
		wangu = B - P;
		wangu1 = Math.abs(B - P);

		emax = A + (1 - A) * wangu1;
		target = 1 - (1 - emax) * Math.pow(t, 3);
		// 例：時間依存の線形な譲歩関数

		shiwang = target - B;
		target1 = 1
				- (1 - emax) * Math.pow(0.8 * t, (3 + 2 * shiwang - 0.8 * t));

		System.out.println(target1);

		// with 50% chance, counter offer
		// if we are the first party, also offer.

		if (getUtility(lastReceivedBid) >= target1
				&& validActions.contains(Accept.class)) {
			return new Accept(getPartyId(), lastReceivedBid);
		}
		Bid newBid = generateRandomBid();
		int i = 999;

		while (i > 0 || lastReceivedBid != null || lastReceivedBid == null) {
			newBid = generateRandomBid();

			if (getUtility(newBid) >= target1)
				break;

			i--;

		}
		return new Offer(getPartyId(), newBid);
	}

	/**
	 * All offers proposed by the other parties will be received as a message.
	 * You can use this information to your advantage, for example to predict
	 * their utility.
	 *
	 * @param sender
	 *            The party that did the action. Can be null.
	 * @param action
	 *            The action that party did.
	 */
	@Override
	public void receiveMessage(AgentID sender, Action action) {
		super.receiveMessage(sender, action);
		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2017";
	}

}
