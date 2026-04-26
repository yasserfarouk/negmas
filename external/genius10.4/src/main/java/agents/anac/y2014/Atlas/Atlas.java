package agents.anac.y2014.Atlas;

import java.io.Serializable;
import java.util.List;
import java.util.Random;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueInteger;
import genius.core.issue.ValueInteger;

//csv�o—Í
//import java.io.FileWriter;
//import java.io.BufferedWriter;
//import java.io.PrintWriter;

public class Atlas extends Agent {
	private Action actionOfPartner = null;

	// Ž©•ª‚Ì�÷•à�í—ª‚ÉŠÖ‚·‚é•Ï�”
	private boolean tension = true; // �÷•à‚ÉŠÖ‚µ‚Ä‚ÌŽp�¨�i�‰Šú’l�F‹­‹C�j�i–{“–‚ÍŽã‹C‚Ì‚Ù‚¤‚ª‚¢‚¢‚ª�C�‰Žè‚ªŽã‹C‚¾‚Æ‘ŠŽè‚ÉŠ¨ˆá‚¢‚ð‚³‚¹‚Ä‚µ‚Ü‚¤‚Æ‚¢‚¤ƒ�ƒ^“I‚È�í—ª�j
	private double border_bid_utility = 1.0; // ‹«ŠE’l

	// ‘ŠŽè‚ÌofferŠÖ˜A‚Ì•Ï�”
	private Bid best_partner_bid = null; // Œ»�Ý‚ÌŒð�Â’†‚É‚¨‚¯‚é‘ŠŽè‚Ì’ñŽ¦Bid‚Ì’†‚Å�Å‘åŒø—p’l‚Æ‚È‚éBid
	private Bid prev_partner_bid = null; // ˆê‚Â‘O‚Ì‘ŠŽè‚Ì’ñŽ¦Bid�i‹ß–T’T�õ‚Å—p‚¢‚é�j
	private double prev_offeredUtilFromOpponent = 0;

	// ”ñ�íŽžŠÖ˜A
	private boolean attack_flag = false;
	private double partner_last_bid_time = 0.0;
	private double partner_prev_last_bid_time = 0.0;
	private double last_time = 0.99;
	private double time_scale = 0.01;

	// ”»’èŠÖ˜A‚Ìƒtƒ‰ƒO
	private boolean first_offer_flag = true;
	private boolean first_offer_decision_flag = false;
	private boolean repeat_flag = true;

	// ‰ß‹Ž‚ÌƒZƒbƒVƒ‡ƒ“ƒf�[ƒ^‚©‚ç’l‚ðŽó‚¯Žæ‚é•Ï�”
	private boolean last_tension = true;
	private boolean is_agreement;
	private Bid last_bid = null;
	private Bid max_bid = null;
	private Bid true_best_agreement_bid = null;
	private Bid false_best_agreement_bid = null;
	private int true_failure_num = 0;
	private int false_agreement_num = 0;
	private double false_agreement_average_utility_with_discount = 0;
	private double partner_attack_num = 0;
	private double partner_guard_num = 0;
	private double last_bid_time = 1.0;
	private boolean tension_true_seal_flag = false;
	private double true_repeat_time = 1.0;
	private double false_repeat_time = 1.0;
	double false_agreement_best_utility = 0.0;

	// �d‚Ý•t‚«•½‹Ï‚ð‹�‚ß‚é‚½‚ß‚É—p‚¢‚é•Ï�”
	private int partner_bid_count = 0;
	private double partner_bid_utility_weighted_average = 0;

	// ’è�”
	private static int NEAR_ITERATION = 100;
	private static int SA_ITERATION = 10;
	private static int SAMPLE_NUM = 10;
	private static int ARRAY_LIM = 1000;
	private static int ERROR_TIME_SCALE_LIM = 3;

	// // csv�o—Í—p•Ï�”
	// // Ž©•ª‚Ì’ñŽ¦‚µ‚½Bid‚ÌŒø—p’l
	// FileWriter fw;
	// PrintWriter pw_my_u;
	// // ‘ŠŽè‚Ì’ñŽ¦‚µ‚½Bid‚ÌŒø—p’l
	// FileWriter fw_p;
	// PrintWriter pw_pa_u;
	// // Ž©•ª‚Ì’ñŽ¦‚µ‚½Bid‚Ì‹«ŠE�ü
	// FileWriter fw_b;
	// PrintWriter pw_my_b;
	// //
	// Ž©•ª‚Ì’ñŽ¦‚µ‚½Bid‚Ì�d‚Ý•t‚«•â�³�Ï‚Ý‹«ŠE�ü
	// FileWriter fw_b_t;
	// PrintWriter pw_my_b_t;
	// // Ž©•ª‚Ì’ñŽ¦‚µ‚½Bid‚Ì•â�³�Ï‚Ý
	// FileWriter fw_b_c;
	// PrintWriter pw_my_b_c;

	@Override
	public void init() {
		try {
			initPrevSessionData();

			// �Å�‰‚Ì‚P‰ñ–Ú‚Ì‚Ý�CSA‚É‚æ‚èŒø—p’l‚Ì�Å‘å’l‚ð‹�‚ß‚é
			if (sessionNr == 0) {
				Bid check_b;
				for (int i = 0; i < utilitySpace.getDomain().getIssues().size()
						* 10; i++) {
					check_b = getBestBidbySA();
					if (max_bid == null
							|| utilitySpace.getUtility(check_b) >= utilitySpace
									.getUtility(max_bid)) {
						max_bid = new Bid(check_b);
					}
					if (utilitySpace.getUtility(max_bid) == 1.0) {
						break;
					}
				}
			}

		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

	@Override
	public String getVersion() {
		return "3.1";
	}

	@Override
	public String getName() {
		return "Atlas";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	@Override
	public Action chooseAction() {
		Action action = null;
		try {
			if (actionOfPartner == null)
				action = chooseBidAction();
			if (actionOfPartner instanceof Offer) {
				Bid partnerBid = ((Offer) actionOfPartner).getBid();

				// �d‚Ý•t‚«•½‹Ï‚ð‹�‚ß‚é
				if (partner_bid_count == 0) {
					partner_bid_utility_weighted_average = utilitySpace
							.getUtility(partnerBid);
				} else {
					partner_bid_utility_weighted_average = (utilitySpace
							.getUtility(partnerBid)
							+ partner_bid_utility_weighted_average) / 2;
				}
				// ‘ŠŽè‚ÌBid‚µ‚½Žž��‚ð‹L˜^‚µ‚Ä‚¨‚­
				partner_prev_last_bid_time = partner_last_bid_time;
				partner_last_bid_time = timeline.getTime();
				time_scale = (((partner_last_bid_time
						- partner_prev_last_bid_time)
						+ time_scale * partner_bid_count)
						/ (partner_bid_count + 1));
				last_time = 1.0 - time_scale * ERROR_TIME_SCALE_LIM;
				partner_bid_count++;

				// get current time
				double time = timeline.getTime();
				double offeredUtilFromOpponent = utilitySpace
						.getUtility(partnerBid);

				// // //�o—Í—p
				// pw_pa_u.print(timeline.getTime());
				// pw_pa_u.print(",");
				// pw_pa_u.print(offeredUtilFromOpponent);
				// pw_pa_u.print(",");
				// pw_pa_u.println();

				action = chooseBidAction();

				prev_partner_bid = new Bid(partnerBid);
				if (best_partner_bid == null) {
					best_partner_bid = new Bid(partnerBid);
				} else if (utilitySpace.getUtility(partnerBid) > utilitySpace
						.getUtility(best_partner_bid)) {
					best_partner_bid = new Bid(partnerBid);
				}

				Bid myBid = ((Offer) action).getBid();
				double myOfferedUtil = utilitySpace.getUtility(myBid);
				// accept under certain circumstances
				if (isAcceptable(partnerBid, offeredUtilFromOpponent,
						myOfferedUtil, time))
					action = new Accept(getAgentID(), partnerBid);
			}
			// if (timeline.getType().equals(Timeline.Type.Time)) {
			// sleep(0.005); // just for fun
			// }
		} catch (Exception e) {
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid()); // best
																	// guess
																	// if
																	// things
																	// go
																	// wrong.
		}
		return action;
	}

	private boolean isAcceptable(Bid partnerBid, double offeredUtilFromOpponent,
			double myOfferedUtil, double time) throws Exception {
		// ‘ŠŽè‚ÌŒø—p’l
		double P = offeredUtilFromOpponent;
		prev_offeredUtilFromOpponent = offeredUtilFromOpponent;

		// ‹«ŠE’l‚Ì�X�V
		getBorderUtility();

		if (P >= border_bid_utility
				&& P >= utilitySpace.getUtility(best_partner_bid)) {
			return true;
		}

		// ‘¦Œˆ”»’è
		if (!tension && false_best_agreement_bid != null
				&& P >= utilitySpace.getUtility(false_best_agreement_bid)) {
			// ‚±‚Ì‚Ü‚ÜŒð�Â‚ð‘±‚¯‚½�ê�‡
			double parameter = partner_guard_num + partner_attack_num;
			// —˜“¾(‘ŠŽè‚ªŽã‹C‚ÌŽž)
			double get_point = utilitySpace.getDiscountFactor()
					- utilitySpace.getUtility(false_best_agreement_bid);
			// ‘¹Ž¸(‘ŠŽè‚ª‹­‹C‚ÌŽž)
			double lost_point = false_agreement_average_utility_with_discount
					- utilitySpace.getUtility(false_best_agreement_bid);

			// Šú‘Ò’l(‘ŠŽè‚ÌŽp�¨‚Í‹­‹CorŽã‹C‚Å‚»‚ê‚¼‚ê‚ÌŠm—¦‚Í‰ß‹Ž‚ÌŒð�Â‚ðŽQ�Æ‚·‚é)
			double mean_point = get_point
					* ((partner_guard_num + 0.5) / (parameter + 1))
					+ lost_point
							* ((partner_attack_num + 0.5) / (parameter + 1));

			if (mean_point < 0) {
				// first_offer_decision_flag = true;
				return true;
			}
		}

		// ”ñ�í—p
		if (!tension && timeline.getTime() > (last_time - time_scale * 0)
				&& P > utilitySpace.getReservationValue()) {
			attack_flag = true;
			return true;
		}

		return false;
	}

	private Action chooseBidAction() {
		Bid nextBid = null;
		try {
			// ‘¦Œˆ”»’èŠÖ˜A‚Ìƒtƒ‰ƒOŠÇ—�
			first_offer_decision_flag = false;

			nextBid = getBid();

			first_offer_flag = false;
		} catch (Exception e) {
			System.out.println("Problem with received bid:" + e.getMessage()
					+ ". cancelling bidding");
		}
		if (nextBid == null)
			return (new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid()));
		return (new Offer(getAgentID(), nextBid));
	}

	private Bid getBid() throws Exception {
		// ”ñ�í—p
		if (!tension) {
			// Žã‹C‚Ì‚Æ‚«�C‘ŠŽè‚Ì�¡‚Ü‚Å‚ÌŒð�Â‚ÌBid‚ðŒð�Â�I—¹�¡‘O‚Å’ñŽ¦‚·‚é
			if (timeline.getTime() > (last_time - time_scale * 2)
					&& best_partner_bid != null
					&& false_best_agreement_bid != null) {
				attack_flag = true;
				// �¬‚³‚¢•û
				if (timeline.getTime() > (last_time - time_scale * 1)) {
					if (utilitySpace
							.getUtility(false_best_agreement_bid) < utilitySpace
									.getUtility(best_partner_bid)) {
						if (utilitySpace.getUtility(
								false_best_agreement_bid) > utilitySpace
										.getReservationValue()) {
							return false_best_agreement_bid;
						}
					} else {
						if (utilitySpace
								.getUtility(best_partner_bid) > utilitySpace
										.getReservationValue()) {
							return best_partner_bid;
						}
					}
				}
				// ‘å‚«‚¢•û
				if (utilitySpace
						.getUtility(false_best_agreement_bid) > utilitySpace
								.getUtility(best_partner_bid)) {
					if (utilitySpace
							.getUtility(false_best_agreement_bid) > utilitySpace
									.getReservationValue()) {
						return false_best_agreement_bid;
					}
				} else {
					if (utilitySpace.getUtility(best_partner_bid) > utilitySpace
							.getReservationValue()) {
						return best_partner_bid;
					}
				}
			}

			if (timeline.getTime() > (last_time - time_scale * 1) && !tension
					&& false_best_agreement_bid == null
					&& best_partner_bid != null
					&& utilitySpace.getUtility(best_partner_bid) > utilitySpace
							.getReservationValue()) {
				attack_flag = true;
				return best_partner_bid;
			}
			if (timeline.getTime() > (last_time - time_scale * 1) && !tension
					&& best_partner_bid == null
					&& false_best_agreement_bid != null
					&& utilitySpace
							.getUtility(false_best_agreement_bid) > utilitySpace
									.getReservationValue()) {
				attack_flag = true;
				return false_best_agreement_bid;
			}
		}

		// ‹«ŠE’l‚Ì�X�V
		getBorderUtility();

		// csv�o—Í—p
		// pw_my_b_c.print(timeline.getTime());
		// pw_my_b_c.print(",");
		// pw_my_b_c.print(border_bid_utility);
		// pw_my_b_c.print(",");
		// pw_my_b_c.println();

		Bid bid = null;
		Bid check_bid = null;

		// ‹ß–T’T�õ(’¼‘O‚Ì‘ŠŽè‚Ì’ñŽ¦Bid‚ðŠî�€)
		if (prev_partner_bid != null || last_bid != null) {
			if (prev_partner_bid != null) {
				bid = new Bid(prev_partner_bid);
			} else {
				bid = new Bid(last_bid);
			}
			// Žp�¨‚Å’T�õ•û–@‚ð•Ï�X
			try {
				for (int i = 0; i < NEAR_ITERATION; i++) {
					if (tension) {
						// ‹­‹C
						bid = getBestBidbyNear(bid);

						if (utilitySpace.getUtility(bid) > border_bid_utility) {
							break;
						}
					} else {
						// Žã‹C
						bid = getBetterBidbyNear(bid);
					}
				}
			} catch (Exception e) {
				System.out.println("Problem with received bid(Near:last):"
						+ e.getMessage() + ". cancelling bidding");
			}
		}

		// ‹ß–T’T�õ‚ÌŒ‹‰Ê�C–ž‘«‚Ì‚¢‚­Bid‚ð“¾‚ç‚ê‚È‚©‚Á‚½�ê�‡�CSA‚ÅBid‚ð’T�õ
		if (bid == null || utilitySpace.getUtility(bid) < border_bid_utility
				|| (tension && utilitySpace.getUtility(bid) < utilitySpace
						.getUtility(max_bid))) {
			try {
				for (int i = 0; i < SA_ITERATION; i++) {
					if (tension) {
						// ‹­‹C
						check_bid = getBestBidbySA();
						if (bid == null || utilitySpace.getUtility(
								check_bid) > utilitySpace.getUtility(bid)) {
							bid = new Bid(check_bid);
						}

						if (utilitySpace.getUtility(bid) == 1.0) {
							break;
						}
					} else {
						// Žã‹C
						check_bid = getBetterBidbySA();
						if (bid == null || (utilitySpace
								.getUtility(bid) < border_bid_utility
								|| (utilitySpace
										.getUtility(check_bid) < utilitySpace
												.getUtility(bid))
										&& utilitySpace.getUtility(
												check_bid) > border_bid_utility)) {
							bid = new Bid(check_bid);
						}
					}
				}
			} catch (Exception e) {
				System.out.println("Problem with received bid(SA):"
						+ e.getMessage() + ". cancelling bidding");
			}
		}

		// �Å‘åBid‚Ì�X�V
		if (bid != null && utilitySpace.getUtility(max_bid) <= utilitySpace
				.getUtility(bid)) {
			max_bid = new Bid(bid);
		}

		// “¾‚ç‚ê‚½Bid‚ª‹«ŠE’l‚æ‚è‚à�¬‚³‚¢�ê�‡�C�Å‘åŒø—p’l‚Æ‚È‚éBid‚ð’ñŽ¦‚·‚é
		if (bid == null || utilitySpace.getUtility(bid) < border_bid_utility
				|| (tension && utilitySpace.getUtility(bid) < utilitySpace
						.getUtility(max_bid))) {
			bid = new Bid(max_bid);
		}

		// ‘¦Œˆ”»’è
		if (!tension && first_offer_flag && false_best_agreement_bid != null) {
			// ‚±‚Ì‚Ü‚ÜŒð�Â‚ð‘±‚¯‚½�ê�‡
			double parameter = partner_guard_num + partner_attack_num;

			// —˜“¾(‘ŠŽè‚ªŽã‹C‚ÌŽž)
			double get_point = utilitySpace.getDiscountFactor()
					- utilitySpace.getUtility(false_best_agreement_bid);
			// ‘¹Ž¸(‘ŠŽè‚ª‹­‹C‚ÌŽž)
			double lost_point = false_agreement_average_utility_with_discount
					- utilitySpace.getUtility(false_best_agreement_bid);

			// Šú‘Ò’l(‘ŠŽè‚ÌŽp�¨‚Í‹­‹CorŽã‹C‚Å‚»‚ê‚¼‚ê‚ÌŠm—¦‚Í‰ß‹Ž‚ÌŒð�Â‚ðŽQ�Æ‚·‚é)
			double mean_point = get_point
					* ((partner_guard_num + 0.5) / (parameter + 1))
					+ lost_point
							* ((partner_attack_num + 0.5) / (parameter + 1));

			if (mean_point < 0) {
				first_offer_decision_flag = true;
				return false_best_agreement_bid;
			}
		}

		if (last_bid != null) {
			// ‹­‹C‚Ì˜A�Ÿ”»’è
			if (true_best_agreement_bid != null && tension
					&& first_offer_flag) {
				return true_best_agreement_bid;
			}
			// �ÄŒ»”»’è
			if (repeat_flag && utilitySpace.getDiscountFactor() < 1.0) {
				// ‹­‹C‚Ì�ÄŒ»”»’è
				if (tension && true_best_agreement_bid != null
						&& true_repeat_time <= timeline.getTime()) {
					repeat_flag = false;
					return true_best_agreement_bid;
				}

				// Žã‹C‚Ì�ÄŒ»”»’è
				if (!tension && false_best_agreement_bid != null
						&& false_repeat_time <= timeline.getTime()) {
					repeat_flag = false;
					return false_best_agreement_bid;
				}
			}
		}

		// //csv�o—Í—p
		// pw_my_u.print(timeline.getTime());
		// pw_my_u.print(",");
		// pw_my_u.print(utilitySpace.getUtility(bid));
		// pw_my_u.print(",");
		// pw_my_u.println();

		return bid;
	}

	// ‹«ŠE’l‚ð�X�V‚·‚éŠÖ�”
	private void getBorderUtility() throws Exception {
		double compensate = 0; // ‹«ŠE’l•â�³—p‚Ì•Ï�”
		double time_compensate = 0; // ŽžŠÔ‚É‚æ‚é‹«ŠE’l•â�³—p‚Ì•Ï�”
		// ‹«ŠE’l‚Ì�X�V
		border_bid_utility = timeline.getTime()
				* (1.0 - utilitySpace.getDiscountFactor())
				+ utilitySpace.getDiscountFactor();

		// Discount‚ª�‚‚¢‚Æ‚«‚É�÷•à‚µ‚·‚¬‚é‚±‚Æ‚ð–h‚®‚½‚ß‚É�CŠeŽp�¨‚Ì1�í–Ú‚ÍŠî–{‹«ŠE’l‚ð‚Æ‚ç‚È‚¢‚æ‚¤‚É‚·‚é
		if ((tension && true_failure_num == 0)
				|| (!tension && false_agreement_num == 0)) {
			border_bid_utility = 1.0;
		}

		// //csv�o—Í—p
		// pw_my_b.print(timeline.getTime());
		// pw_my_b.print(",");
		// pw_my_b.print(border_bid_utility);
		// pw_my_b.print(",");
		// pw_my_b.println();

		if (tension) {
			// ‚à‚µ�C‹«ŠE�ü‚ª�Å—Ç’ñŽ¦Bid‚ÌŒø—p’l‚æ‚è’á‚¢Žž�C‹«ŠE’l‚ð‚»‚ÌŒø—p’l‚É‚·‚é
			if (true_best_agreement_bid != null
					&& border_bid_utility < utilitySpace
							.getUtility(true_best_agreement_bid)) {
				border_bid_utility = utilitySpace
						.getUtility(true_best_agreement_bid);
			}
		} else {
			// Žã‹C‚Ì�ê�‡
			// �d‚Ý•t‚«•½‹Ï‚ð—˜—p‚µ‚½•â�³‚ð“K—p
			// SAMPLE_NUM(“_)ƒTƒ“ƒvƒ‹�W‚Ü‚Á‚Ä‚¢‚é�ê�‡�C–â‘è‹óŠÔ‚É‚æ‚Á‚Ä‹«ŠE’l‚ðŠÉ˜a‚ð�l‚¦‚é
			if (SAMPLE_NUM < partner_bid_count) {
				compensate = (prev_offeredUtilFromOpponent
						- partner_bid_utility_weighted_average)
						* prev_offeredUtilFromOpponent
						* (1.0 - utilitySpace.getReservationValue());

				// ŽžŠÔŒo‰ß‚Æ‘ŠŽè‚Ì�Å—ÇBid‚É‚æ‚é�÷•à‚ð—˜—p‚µ‚½•â�³‚ð“K—p
				if (best_partner_bid != null) {
					time_compensate = timeline.getTime()
							* (border_bid_utility
									- utilitySpace.getUtility(best_partner_bid))
							* utilitySpace.getUtility(best_partner_bid)
							* (1.0 - utilitySpace.getReservationValue());

					if (time_compensate < 0) {
						time_compensate = 0;
					}
				}

				if (compensate * (1.0 + time_compensate) > time_compensate
						&& false_agreement_num > 0) {
					border_bid_utility = border_bid_utility
							- compensate * (1.0 + time_compensate);
				} else {
					border_bid_utility = border_bid_utility - time_compensate;
				}

				if (timeline.getTime() > 0.90) {
					border_bid_utility = border_bid_utility
							- 10 * (timeline.getTime() - 0.90)
									* (border_bid_utility - utilitySpace
											.getUtility(best_partner_bid));
				}
			}
		}

		// ‚à‚µ�C‹«ŠE�ü‚ª�Å—Ç’ñŽ¦Bid‚ÌŒø—p’l‚æ‚è’á‚¢Žž�C‹«ŠE’l‚ð‚»‚ÌŒø—p’l‚É‚·‚é
		if (false_best_agreement_bid != null
				&& border_bid_utility < utilitySpace
						.getUtility(false_best_agreement_bid)) {
			border_bid_utility = utilitySpace
					.getUtility(false_best_agreement_bid);
		}

		// ‚à‚µ�C‹«ŠE�ü‚ªŒ»�Ý‚ÌŒð�Â‚Ì�Å—Ç’ñŽ¦Bid‚ÌŒø—p’l‚æ‚è’á‚¢Žž�C‹«ŠE’l‚ð‚»‚ÌŒø—p’l‚É‚·‚é
		if (best_partner_bid != null && border_bid_utility < utilitySpace
				.getUtility(best_partner_bid)) {
			border_bid_utility = utilitySpace.getUtility(best_partner_bid);
		}

		// ReservationValue‚Ì•û‚ª‹«ŠE’l‚æ‚è‘å‚«‚­‚È‚éŽž(•ÛŒ¯—p)
		if (border_bid_utility < utilitySpace.getReservationValue()) {
			border_bid_utility = utilitySpace.getReservationValue();
		}

		// //csv�o—Í—p
		// pw_my_b_t.print(timeline.getTime());
		// pw_my_b_t.print(",");
		// pw_my_b_t.print(border_bid_utility);
		// pw_my_b_t.print(",");
		// pw_my_b_t.println();
	}

	// SA(�Å‘åŒø—p’lBid)
	private Bid getBestBidbySA() throws Exception {
		// SA‚É‚æ‚é’T�õ‚ð�s‚¢�CBid‚ð•Ô‚·
		Bid current_b, next_b, bid = null; // Œ»�Ý‚ÌBid‚Æ�Å“KBid‚Æ‹ß–T‚ÌBid�CŽã‹C‚Ì‚Æ‚«‚Ì‚½‚ß‚ÌBid�C•Ô‹p—p‚ÌBid
		double current_u, best_u, next_u; // Œ»�Ý‚ÌBid‚ÌŒø—p’l‚Æ�Å“KBid‚ÌŒø—p’l‚Æ‹ß–T‚ÌBid‚ÌŒø—p’l
		int next_v; // ‹ß–T‚ÌBid‚Ì•Ï�XŒã‚Ìvalue‚Ì’l
		double temperature; // ‰·“x
		double cool = 0.99; // —â‹p“x
		double p; // ‘JˆÚŠm—¦
		double diff; // Œ»�Ý‚ÌBid‚Æ‹ß–T‚ÌBid‚ÌŒø—p’l‚Ì�·
		Random randomnr = new Random(); // —��”
		int step = 1;// •Ï�X‚·‚é•�
		int step_num = 1; // •Ï�X‚·‚é‰ñ�”
		int flag; // step‚Ì•„�†

		Bid best[];
		best = new Bid[ARRAY_LIM];
		int random_count = 0; // “¯‚¶Œø—p’l‚ðŽ�‚Â�Å‘å’l‚ÌBid‚ÌŒÂ�”

		// �‰Šú‰ð‚Ì�¶�¬
		if (prev_partner_bid == null) {
			current_b = utilitySpace.getDomain().getRandomBid(null);
		} else {
			current_b = new Bid(prev_partner_bid);
		}

		do {
			current_u = utilitySpace.getUtility(current_b);

			// �Å“K‰ð‚Ì�‰Šú‰»
			best[0] = new Bid(current_b);
			best_u = current_u;
			// ƒJƒEƒ“ƒ^‚ÌƒŠƒZƒbƒg
			random_count = 0;

			temperature = 1000000; // ŠJŽn‰·“x‚ÌƒZƒbƒg

			// ‰·“x‚ª�\•ª‰º‚ª‚é‚©�C–Ú“I‚Ìƒ†�[ƒeƒBƒŠƒeƒB‚ðŽæ“¾‚Å‚«‚é‚Ü‚Åƒ‹�[ƒv
			while (temperature > 0.0001) {
				next_b = new Bid(current_b); // next_b‚ð�‰Šú‰»
				List<Issue> issues = utilitySpace.getDomain().getIssues(); // ‘Sissue‚ÌŽæ“¾

				// ‹ß–T‚ÌBid‚ðŽæ“¾‚·‚é
				for (int i = 0; i < step_num; i++) {
					int issueIndex = randomnr.nextInt(issues.size()); // issue‚Ì”ÍˆÍ“à‚Åindex‚ðƒ‰ƒ“ƒ_ƒ€‚ÉŽw’è
					IssueInteger lIssueInteger = (IssueInteger) issues
							.get(issueIndex); // Žw’è‚µ‚½index‚Ìissue
					int issueNumber = lIssueInteger.getNumber(); // issue”Ô�†
					int max = lIssueInteger.getUpperBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å‘å’l
					int min = lIssueInteger.getLowerBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å�¬’l
					ValueInteger issueValue = (ValueInteger) next_b
							.getValue(issueNumber); // ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue
					int issueValueInt = Integer.valueOf(issueValue.toString())
							.intValue(); // ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue(integer)

					// velue‚Ì�ãŒÀ‚Æ‰ºŒÀ‚ð�l‚¦‚Ä•ÏˆÊ�istep�j‚Ì•„�†(flag)‚ð�Ý’è‚·‚é
					if (issueValueInt + step > max) {
						flag = -1;
					} else if (issueValueInt - step < min) {
						flag = 1;
					} else if (randomnr.nextInt(2) == 0) {
						flag = 1;
					} else {
						flag = -1;
					}

					next_v = issueValueInt + flag * step; // •Ï�XŒã‚Ìvalue‚ðŒvŽZ‚·‚é
					next_b = next_b.putValue(issueNumber,
							new ValueInteger(next_v)); // ‹ß–T‚ÌBid‚ð‹�‚ß‚é
				}

				next_u = utilitySpace.getUtility(next_b); // ‹ß–T‚ÌBid‚ÌŒø—p’l‚ð‹�‚ß‚é
				diff = current_u - next_u; // ‘JˆÚŠm—¦‚ð‹�‚ß‚é

				if (diff > 0.0) {
					p = Math.exp(-diff / temperature); // Œ»�Ý‚ÌŒø—p’l‚Ì•û‚ª�‚‚¢�ê�‡
				} else {
					p = 1.0; // ‹ß–T‚ÌŒø—p’l‚Ì•û‚ª�‚‚¢�ê�‡
				}

				if (p > randomnr.nextDouble()) {
					current_b = new Bid(next_b); // Bid‚Ì�X�V
					current_u = next_u; // Utility‚Ì�X�V
				}

				// �Å“K‰ð‚Ì�X�V
				if (next_u > best_u) {
					random_count = 0;
					best[random_count] = new Bid(next_b);
					best_u = next_u;
					// ƒJƒEƒ“ƒ^‚ÌƒŠƒZƒbƒg
					random_count++;
				}

				// “¯‚¶Œø—p’l‚ðŽ�‚ÂBid‚ð‹L‰¯‚µ‚Ä‚¨‚­
				if (next_u == best_u) {
					best[random_count % ARRAY_LIM] = new Bid(next_b);
					random_count++;
				}
				// ‰·“x‚ð‰º‚°‚é
				temperature = temperature * cool;
			}

			// Œø—p’l‚ª�Å‘å‚Æ‚È‚éBid‚ð•Ô‚·
			if (bid == null || best_u > utilitySpace.getUtility(bid)) {
				if (random_count > ARRAY_LIM) {
					bid = new Bid(best[randomnr.nextInt(ARRAY_LIM)]);
				} else {
					bid = new Bid(best[randomnr.nextInt(random_count)]);
				}
			}
		} while (utilitySpace.getUtility(bid) < utilitySpace
				.getReservationValue());

		return bid;
	}

	// SA(‹«ŠE’l•t‹ß‚ÌBid)
	private Bid getBetterBidbySA() throws Exception {
		// SA‚É‚æ‚é’T�õ‚ð�s‚¢�CBid‚ð•Ô‚·
		Bid current_b, best_b, next_b, better_b = null, bid = null; // Œ»�Ý‚ÌBid‚Æ�Å“KBid‚Æ‹ß–T‚ÌBid�CŽã‹C‚Ì‚Æ‚«‚Ì‚½‚ß‚ÌBid�C•Ô‹p—p‚ÌBid
		double current_u, best_u, next_u, better_u = 0; // Œ»�Ý‚ÌBid‚ÌŒø—p’l‚Æ�Å“KBid‚ÌŒø—p’l‚Æ‹ß–T‚ÌBid‚ÌŒø—p’l
		int next_v; // ‹ß–T‚ÌBid‚Ì•Ï�XŒã‚Ìvalue‚Ì’l
		double temperature; // ‰·“x
		double cool = 0.99; // —â‹p“x
		double p; // ‘JˆÚŠm—¦
		double diff; // Œ»�Ý‚ÌBid‚Æ‹ß–T‚ÌBid‚ÌŒø—p’l‚Ì�·
		Random randomnr = new Random(); // —��”
		int step = 1;// •Ï�X‚·‚é•�
		int step_num = 1; // •Ï�X‚·‚é‰ñ�”
		int flag; // step‚Ì•„�†

		Bid better[];
		better = new Bid[ARRAY_LIM];
		int random_count = 0; // “¯‚¶Œø—p’l‚ðŽ�‚Â‹«ŠE’l•t‹ß‚ÌBid‚ÌŒÂ�”

		// �‰Šú‰ð‚Ì�¶�¬
		if (prev_partner_bid == null) {
			current_b = utilitySpace.getDomain().getRandomBid(null);
		} else {
			current_b = new Bid(prev_partner_bid);
		}

		do {
			current_u = utilitySpace.getUtility(current_b);

			// �Å“K‰ð‚Ì�‰Šú‰»
			best_b = new Bid(current_b);
			best_u = current_u;

			// ‹«ŠE’l•t‹ß‚ÌBid‚Ì�‰Šú‰»
			better_b = null;
			better_u = 0;

			temperature = 1000000; // ŠJŽn‰·“x‚ÌƒZƒbƒg

			// ‰·“x‚ª�\•ª‰º‚ª‚é‚©�C–Ú“I‚Ìƒ†�[ƒeƒBƒŠƒeƒB‚ðŽæ“¾‚Å‚«‚é‚Ü‚Åƒ‹�[ƒv
			while (temperature > 0.0001) {
				next_b = new Bid(current_b); // next_b‚ð�‰Šú‰»
				List<Issue> issues = utilitySpace.getDomain().getIssues(); // ‘Sissue‚ÌŽæ“¾

				// ‹ß–T‚ÌBid‚ðŽæ“¾‚·‚é
				for (int i = 0; i < step_num; i++) {
					int issueIndex = randomnr.nextInt(issues.size()); // issue‚Ì”ÍˆÍ“à‚Åindex‚ðƒ‰ƒ“ƒ_ƒ€‚ÉŽw’è
					IssueInteger lIssueInteger = (IssueInteger) issues
							.get(issueIndex); // Žw’è‚µ‚½index‚Ìissue
					int issueNumber = lIssueInteger.getNumber(); // issue”Ô�†
					int max = lIssueInteger.getUpperBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å‘å’l
					int min = lIssueInteger.getLowerBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å�¬’l
					ValueInteger issueValue = (ValueInteger) next_b
							.getValue(issueNumber); // ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue
					int issueValueInt = Integer.valueOf(issueValue.toString())
							.intValue(); // ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue(integer)

					// velue‚Ì�ãŒÀ‚Æ‰ºŒÀ‚ð�l‚¦‚Ä•ÏˆÊ�istep�j‚Ì•„�†(flag)‚ð�Ý’è‚·‚é
					if (issueValueInt + step > max) {
						flag = -1;
					} else if (issueValueInt - step < min) {
						flag = 1;
					} else if (randomnr.nextInt(2) == 0) {
						flag = 1;
					} else {
						flag = -1;
					}

					next_v = issueValueInt + flag * step; // •Ï�XŒã‚Ìvalue‚ðŒvŽZ‚·‚é
					next_b = next_b.putValue(issueNumber,
							new ValueInteger(next_v)); // ‹ß–T‚ÌBid‚ð‹�‚ß‚é
				}

				next_u = utilitySpace.getUtility(next_b); // ‹ß–T‚ÌBid‚ÌŒø—p’l‚ð‹�‚ß‚é
				diff = current_u - next_u; // ‘JˆÚŠm—¦‚ð‹�‚ß‚é

				if (diff > 0.0) {
					p = Math.exp(-diff / temperature); // Œ»�Ý‚ÌŒø—p’l‚Ì•û‚ª�‚‚¢�ê�‡
				} else {
					p = 1.0; // ‹ß–T‚ÌŒø—p’l‚Ì•û‚ª�‚‚¢�ê�‡
				}

				if (p > randomnr.nextDouble()) {
					current_b = new Bid(next_b); // Bid‚Ì�X�V
					current_u = next_u; // Utility‚Ì�X�V
				}

				// �Å“K‰ð‚Ì�X�V
				if (next_u > best_u) {
					best_b = new Bid(next_b);
					best_u = next_u;
				}

				// ‹«ŠE’l•t‹ß‚Ì’l‚ð•Ô‚·‚½‚ß‚É‹L‰¯‚·‚é
				// �‰Šú‰»
				if (better_b == null && best_u > border_bid_utility) {
					better_b = new Bid(best_b);
					better_u = best_u;
					random_count = 0;
					better[random_count] = new Bid(next_b);
					// ƒJƒEƒ“ƒ^‚ðƒZƒbƒg
					random_count++;
				}
				// �X�V
				if (next_u < better_u && next_u > border_bid_utility) {
					better_b = new Bid(next_b);
					better_u = next_u;
					random_count = 0;

					better[random_count] = new Bid(next_b);
					// ƒJƒEƒ“ƒ^‚ðƒZƒbƒg
					random_count++;
				}

				// “¯‚¶Œø—p’l‚ðŽ�‚ÂBid‚ð‹L‰¯‚µ‚Ä‚¨‚­
				if (next_u == better_u) {
					better[random_count % ARRAY_LIM] = new Bid(next_b);
					random_count++;
				}

				// ‰·“x‚ð‰º‚°‚é
				temperature = temperature * cool;
			}

			if (better_b != null) {
				// Œø—p’l‚ª‹«ŠE’l•t‹ß‚Æ‚È‚éBid‚ð•Ô‚·
				if (bid == null || better_u < utilitySpace.getUtility(bid)) {
					if (random_count > ARRAY_LIM) {
						bid = new Bid(better[randomnr.nextInt(ARRAY_LIM)]);
					} else {
						bid = new Bid(better[randomnr.nextInt(random_count)]);
					}
				}
			} else {
				// ‹«ŠE’l‚æ‚è‘å‚«‚ÈŒø—p’l‚ðŽ�‚ÂBid‚ªŒ©‚Â‚©‚ç‚È‚©‚Á‚½‚Æ‚«‚Í�CŒø—p’l‚ª�Å‘å‚Æ‚È‚éBid‚ð•Ô‚·
				if (bid == null) {
					bid = new Bid(best_b);
				}
			}
		} while (utilitySpace.getUtility(bid) < utilitySpace
				.getReservationValue());

		return bid;
	}

	// ‹ß–T’T�õ(�Å‘åŒø—p’lBid)
	private Bid getBestBidbyNear(Bid baseBid) throws Exception {
		Bid current_b = new Bid(baseBid); // Œ»�Ý‚ÌBid
		double current_u = utilitySpace.getUtility(baseBid); // Œ»�Ý‚ÌBid‚ÌŒø—p’l

		Bid best_b = new Bid(baseBid); // �Å“KBid
		double best_u = current_u; // �Å“KBid‚ÌŒø—p’l

		Random randomnr = new Random(); // —��”
		Bid best[];
		best = new Bid[ARRAY_LIM];
		int random_count = 0; // “¯‚¶Œø—p’l‚ðŽ�‚Â�Å‘å’l‚ÌBid‚ÌŒÂ�”

		List<Issue> issues = utilitySpace.getDomain().getIssues(); // ‘Sissue‚ÌŽæ“¾

		int lim = issues.size();

		for (int i = 0; i < lim; i++) {
			IssueInteger lIssueInteger = (IssueInteger) issues.get(i); // Žw’è‚µ‚½index‚Ìissue
			int issueNumber = lIssueInteger.getNumber(); // issue”Ô�†
			int max = lIssueInteger.getUpperBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å‘å’l
			int min = lIssueInteger.getLowerBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å�¬’l

			for (int j = min; j <= max; j++) {
				current_b = current_b.putValue(issueNumber,
						new ValueInteger(j)); // ‹ß–T‚ÌBid‚ð‹�‚ß‚é
				current_u = utilitySpace.getUtility(current_b);

				// �Å‘åŒø—p’l‚Æ‚È‚éBid‚ð•Û‘¶
				if (current_u > best_u) {
					best_b = new Bid(current_b);
					best_u = utilitySpace.getUtility(current_b);
					random_count = 0;
					best[random_count] = new Bid(current_b);
					// ƒJƒEƒ“ƒ^‚ÌƒŠƒZƒbƒg
					random_count++;
				}

				// “¯‚¶Œø—p’l‚ðŽ�‚ÂBid‚ð‹L‰¯‚µ‚Ä‚¨‚­
				if (current_u == best_u) {
					best[random_count % ARRAY_LIM] = new Bid(current_b);
					random_count++;
				}

				current_b = new Bid(baseBid);
			}
		}

		// Œø—p’l‚ª�Å‘å‚Æ‚È‚éBid‚ð•Ô‚·
		if (random_count > ARRAY_LIM) {
			best_b = new Bid(best[randomnr.nextInt(ARRAY_LIM)]);
		} else {
			best_b = new Bid(best[randomnr.nextInt(random_count)]);
		}

		return best_b;
	}

	// ‹ß–T’T�õ(‹«ŠE’l•t‹ß‚ÌBid)
	private Bid getBetterBidbyNear(Bid baseBid) throws Exception {
		Bid current_b = new Bid(baseBid); // Œ»�Ý‚ÌBid
		double current_u = utilitySpace.getUtility(baseBid); // Œ»�Ý‚ÌBid‚ÌŒø—p’l

		Bid best_b = new Bid(baseBid); // �Å“KBid
		double best_u = current_u; // �Å“KBid‚ÌŒø—p’l

		Bid better_b = null; // ‹«ŠE’l•t‹ß‚ÌŒø—p’l‚ðŽ�‚ÂBid
		double better_u = 0; // ‹«ŠE’l•t‹ß‚ÌŒø—p’l‚ðŽ�‚ÂBid‚ÌŒø—p’l

		Random randomnr = new Random(); // —��”
		Bid better[];
		better = new Bid[ARRAY_LIM];
		int random_count = 0; // “¯‚¶Œø—p’l‚ðŽ�‚Â‹«ŠE’l•t‹ß‚ÌBid‚ÌŒÂ�”

		List<Issue> issues = utilitySpace.getDomain().getIssues(); // ‘Sissue‚ÌŽæ“¾

		int lim = issues.size();

		for (int i = 0; i < lim; i++) {
			IssueInteger lIssueInteger = (IssueInteger) issues.get(i); // Žw’è‚µ‚½index‚Ìissue
			int issueNumber = lIssueInteger.getNumber(); // issue”Ô�†
			int max = lIssueInteger.getUpperBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å‘å’l
			int min = lIssueInteger.getLowerBound();// ‘I‚ñ‚¾issue‚É‘Î‰ž‚·‚évalue‚Ì�Å�¬’l

			for (int j = min; j <= max; j++) {
				current_b = current_b.putValue(issueNumber,
						new ValueInteger(j)); // ‹ß–T‚ÌBid‚ð‹�‚ß‚é
				current_u = utilitySpace.getUtility(current_b);

				// �Å‘åŒø—p’l‚Æ‚È‚éBid‚ð•Û‘¶
				if (current_u > best_u) {
					best_b = new Bid(current_b);
					best_u = utilitySpace.getUtility(current_b);
				}

				// ‹«ŠE’l•t‹ß‚Ì’l‚ð•Ô‚·‚½‚ß‚É‹L‰¯‚·‚é
				// �‰Šú‰»
				if (better_b == null && best_u > border_bid_utility) {
					better_b = new Bid(best_b);
					better_u = current_u;

					better[0] = new Bid(best_b);
					// ƒJƒEƒ“ƒ^‚ðƒZƒbƒg
					random_count = 1;
				}
				// �X�V
				if (current_u < better_u && current_u > border_bid_utility) {
					better_b = new Bid(current_b);
					better_u = current_u;
					random_count = 0;
					better[random_count] = new Bid(current_b);
					// ƒJƒEƒ“ƒ^‚ðƒZƒbƒg
					random_count++;
				}

				// “¯‚¶Œø—p’l‚ðŽ�‚ÂBid‚ð‹L‰¯‚µ‚Ä‚¨‚­
				if (current_u == better_u) {
					better[random_count % ARRAY_LIM] = new Bid(current_b);
					random_count++;
				}

				current_b = new Bid(baseBid);
			}
		}

		if (better_b != null) {
			// Œø—p’l‚ª‹«ŠE’l•t‹ß‚Æ‚È‚éBid‚ð•Ô‚·
			if (random_count > ARRAY_LIM) {
				better_b = new Bid(better[randomnr.nextInt(ARRAY_LIM)]);
			} else {
				better_b = new Bid(better[randomnr.nextInt(random_count)]);
			}
		} else {
			// ‹«ŠE’l‚æ‚è‘å‚«‚ÈŒø—p’l‚ðŽ�‚ÂBid‚ªŒ©‚Â‚©‚ç‚È‚©‚Á‚½‚Æ‚«‚Í�CŒø—p’l‚ª�Å‘å‚Æ‚È‚éBid‚ð•Ô‚·
			better_b = new Bid(best_b);
		}

		return better_b;
	}

	private void initPrevSessionData() throws Exception {
		// csv�o—Í—p
		// fw = new FileWriter("/Users/Mori/Program/R/anac/Atlas_Session_mu"+
		// sessionNr + ".csv", false);
		// pw_my_u = new PrintWriter(new BufferedWriter(fw));
		// // ‘ŠŽè‚Ì’ñŽ¦‚µ‚½Bid‚ÌŒø—p’l
		// fw_p = new
		// FileWriter("/Users/Mori/Program/R/anac/Atlas_Session_Data_pu" +
		// sessionNr + ".csv", false);
		// pw_pa_u = new PrintWriter(new BufferedWriter(fw_p));
		// // Ž©•ª‚Ì’ñŽ¦‚µ‚½Bid‚Ì‹«ŠE�ü
		// fw_b = new
		// FileWriter("/Users/Mori/Program/R/anac/Atlas_Session_Data_mb" +
		// sessionNr + ".csv", false);
		// pw_my_b = new PrintWriter(new BufferedWriter(fw_b));
		// //
		// Ž©•ª‚Ì’ñŽ¦‚µ‚½Bid‚Ì�d‚Ý•t‚«•â�³�Ï‚Ý‹«ŠE�ü
		// fw_b_t = new
		// FileWriter("/Users/Mori/Program/R/anac/Atlas_Session_Data_mbt" +
		// sessionNr + ".csv", false);
		// pw_my_b_t = new PrintWriter(new BufferedWriter(fw_b_t));
		// // Ž©•ª‚Ì’ñŽ¦‚µ‚½Bid‚Ì•â�³�Ï‚Ý
		// fw_b_c = new
		// FileWriter("/Users/Mori/Program/R/anac/Atlas_Session_Data_mbc" +
		// sessionNr + ".csv", false);
		// pw_my_b_c = new PrintWriter(new BufferedWriter(fw_b_c));

		Serializable prev = this.loadSessionData();
		MySessionData mySessionData = (MySessionData) prev;

		// •Ï�”‚Ì�‰Šú‰»
		if (sessionNr > 0 && prev != null) {
			mySessionData = (MySessionData) prev;
			last_tension = mySessionData.tension;
			is_agreement = mySessionData.isAgreement;
			last_bid = mySessionData.lastBid;
			max_bid = mySessionData.max_bid;
			true_best_agreement_bid = mySessionData.true_best_agreement_bid;
			false_best_agreement_bid = mySessionData.false_best_agreement_bid;
			true_failure_num = mySessionData.true_failure_num;
			false_agreement_num = mySessionData.false_agreement_num;
			false_agreement_average_utility_with_discount = mySessionData.false_agreement_average_utility_with_discount;
			partner_attack_num = mySessionData.partner_attack_num;
			partner_guard_num = mySessionData.partner_guard_num;
			last_time = mySessionData.last_time;
			last_bid_time = mySessionData.last_bid_time;
			tension_true_seal_flag = mySessionData.tension_true_seal_flag;
			true_repeat_time = mySessionData.true_repeat_time;
			false_repeat_time = mySessionData.false_repeat_time;
			false_agreement_best_utility = mySessionData.false_agreement_best_utility;

			// Žp�¨‚ÌŒˆ’è
			if (is_agreement && last_tension) {
				// ‹­‹C‚Å�‡ˆÓ‚µ‚½�ê�‡�C‹­‹C‚ÅŒð�Â‚ð‘±�s‚·‚é
				tension = true;
			} else {
				if (false_best_agreement_bid == null) {
					tension = false;
				} else {
					// ‹­‹C‚Ì�ê�‡
					double parameter = partner_guard_num + partner_attack_num;
					if (parameter == 0) {
						parameter = 1;
					}
					// —˜“¾�i‘ŠŽè‚ªŽã‹C‚Ì‚Æ‚«‚É“¾‚éŒø—p’l�iŽc‚è‚ÌŒð�Â‰ñ�”•ª�j�j
					double get_utility_true = (utilitySpace.getDiscountFactor()
							- false_agreement_average_utility_with_discount)
							* (sessionsTotal - (sessionNr));
					// ‘¹Ž¸�i‹­‹C‚ÅŽ¸”s‚µ‚½‚Æ‚«‚ÉŽ¸‚¤Œø—p’l�i1‰ñ•ª�j�j
					double lost_utility_true = utilitySpace
							.getReservationValueWithDiscount(1.0)
							- false_agreement_average_utility_with_discount;
					// Šú‘Ò’l(‘ŠŽè‚ÌŽp�¨‚Í‹­‹CorŽã‹C‚Å‚»‚ê‚¼‚ê‚ÌŠm—¦‚Í‰ß‹Ž‚ÌŒð�Â‚ðŽQ�Æ‚·‚é)
					double utility_true = get_utility_true
							* ((partner_guard_num) / (parameter))
							+ lost_utility_true
									* ((partner_attack_num) / (parameter));
					// Šú‘Ò’l‚É‹­‹C‚Å‚ÌŽ¸”s‚ÌŽÀ�Ñ’l‚ð”½‰f
					utility_true += (utilitySpace
							.getReservationValueWithDiscount(1.0)
							- false_agreement_average_utility_with_discount)
							* true_failure_num;

					if (utility_true > 0) {
						tension = true;
					} else {
						tension = false;
					}
				}
			}

			// 2�í–Ú‚ÍDiscountFactor‚ª1.0‚æ‚è�¬‚³‚¢Žž�C•K‚¸Žã‹C‚É‚È‚é
			if (sessionNr == 1 && utilitySpace.getDiscountFactor() < 1.0) {
				tension = false;
			}

			// 3�í–ÚˆÈ�~‚Í1�í–Ú‚Å�‡ˆÓ‚É�¬Œ÷‚µ‚Ä‚¢‚é‚Æ‚«‹­‹C‚É–ß‚·
			if (sessionNr > 1 && utilitySpace.getDiscountFactor() < 1.0
					&& true_failure_num == 0) {
				tension = true;
			}

			// ‹­‹C••ˆóƒtƒ‰ƒO‚ª—§‚Á‚Ä‚¢‚éŽž‚ÍŽã‹CŒÅ’è
			if (tension_true_seal_flag) {
				tension = false;
			}
		}
	}

	@Override
	public void endSession(NegotiationResult result) {
		try {
			// data‚ðcsv�o—Í‚·‚éŠÖ�”
			// pw_my_u.close();
			// pw_pa_u.close();
			// pw_my_b.close();
			// pw_my_b_t.close();
			// pw_my_b_c.close();

			boolean isAgreement = result.isAgreement();
			Bid lastBid = null;

			if (isAgreement) {
				// Accept‚µ‚½�ê�‡
				lastBid = result.getLastBid();

				if (tension) {
					// ‹­‹C‚ÌŽž
					partner_guard_num++;

					// �‡ˆÓ‚µ‚½�Å‘åŒø—p’lBid‚ð‹L˜^
					if (true_best_agreement_bid == null
							|| utilitySpace.getUtility(lastBid) >= utilitySpace
									.getUtility(true_best_agreement_bid)) {
						true_best_agreement_bid = new Bid(lastBid);
						true_repeat_time = timeline.getTime();
					}
				} else {
					// ‘¦Œˆ‚µ‚½�ê�‡‚ð�œ‚­�ˆ—�
					if (!first_offer_decision_flag) {
						// Žã‹C‚ÌŽž
						if (attack_flag) {
							partner_attack_num++;
						} else {
							if (false_best_agreement_bid != null) {
								// “¾‚ç‚ê‚½Œø—p’l‚ª�C‰ß‹Ž‚ÌŒð�Â‚Ì�Å—Ç‚ÌŒø—p’lˆÈ‰º‚Å‚ ‚è�C‚©‚Â�C‘O‰ñ‚ÌŽp�¨‚Æ�¡‰ñ‚ÌŽp�¨‚ª“¯‚¶‚Å‚ ‚é�ê�‡
								if (utilitySpace.getUtility(
										lastBid) < utilitySpace.getUtility(
												false_best_agreement_bid)
										&& ((tension && last_tension)
												|| (!tension
														&& !last_tension))) {
									partner_attack_num++;
								}
								// “¾‚ç‚ê‚½Œø—p’l‚ª�C‰ß‹Ž‚ÌŒð�Â‚Ì�Å—Ç‚ÌŒø—p’lˆÈ�ã‚Å‚ ‚è�C‚©‚Â�C‘O‰ñ‚ÌŽp�¨‚Æ�¡‰ñ‚ÌŽp�¨‚ª“¯‚¶‚Å‚ ‚é�ê�‡
								if (utilitySpace.getUtility(
										lastBid) > utilitySpace.getUtility(
												false_best_agreement_bid)
										&& ((tension && last_tension)
												|| (!tension
														&& !last_tension))) {
									partner_guard_num++;
								}
							}
						}

						// �‡ˆÓ‚µ‚½�Å‘åŒø—p’lBid‚ð‹L˜^
						if (false_best_agreement_bid == null || utilitySpace
								.getUtility(lastBid) >= utilitySpace
										.getUtility(false_best_agreement_bid)) {
							false_best_agreement_bid = new Bid(lastBid);
							// �ÄŒ»ŽžŠÔ‚ð�Ý’è‚·‚é
							false_repeat_time = timeline.getTime();
						}

						// �‡ˆÓ‚µ‚½Discount�ž‚Ý�Å‘åŒø—p’l‚ð‹L˜^
						if (utilitySpace.getUtilityWithDiscount(lastBid,
								timeline.getTime()) > false_agreement_best_utility) {
							false_agreement_best_utility = utilitySpace
									.getUtilityWithDiscount(lastBid,
											timeline.getTime());
						}

						// •½‹Ï’l‚ð‹L˜^
						false_agreement_average_utility_with_discount = ((false_agreement_average_utility_with_discount
								* false_agreement_num)
								+ utilitySpace.getUtilityWithDiscount(lastBid,
										timeline.getTime()))
								/ (false_agreement_num + 1);
						false_agreement_num++;

						// ‹­‹C••ˆóƒtƒ‰ƒO‚ÌŠm’è
						if (utilitySpace.getDiscountFactor() < utilitySpace
								.getUtilityWithDiscount(lastBid,
										timeline.getTime())
								&& utilitySpace.getDiscountFactor() < 1.0
								&& utilitySpace.getUtility(
										false_best_agreement_bid) < 1.0) {
							tension_true_seal_flag = true;
						} else {
							tension_true_seal_flag = false;
						}
					}
				}
			} else {
				// Accept‚µ‚È‚©‚Á‚½�ê�‡
				partner_attack_num++;

				if (tension) {
					// ‹­‹C‚ÌŽž
					true_failure_num++;
				} else {
					// Žã‹C‚ÌŽž
					// Šî–{“I‚ÉŒð�Â‚Í�¬—§‚·‚é‚ª�C‘ŠŽè‚ª‚ ‚Ü‚è‚É‚à’x‚¢‚Æ‚«‚É‚ÍŒð�Â‚ª�¬—§‚µ‚È‚¢�ê�‡‚ª‚ ‚é‚Ì‚ÅƒAƒWƒƒƒXƒg�D
					last_time = partner_last_bid_time;
				}
			}

			// ‘Å‚¿�Ø‚èŽž��‚Ì�Ý’è
			if (last_time < 1.0 - time_scale || sessionNr == 0) {
				last_time = 1.0 - time_scale;
			}

			MySessionData mySessionData = new MySessionData(tension,
					isAgreement, lastBid, max_bid, true_best_agreement_bid,
					false_best_agreement_bid, true_failure_num,
					false_agreement_num,
					false_agreement_average_utility_with_discount,
					partner_attack_num, partner_guard_num, last_time,
					timeline.getTime(), tension_true_seal_flag,
					true_repeat_time, false_repeat_time,
					false_agreement_best_utility);
			this.saveSessionData(mySessionData);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}
}

class MySessionData implements Serializable {
	Boolean tension; // Œð�Â’†‚ÌŽp�¨
	Boolean isAgreement; // �‡ˆÓ‚µ‚½‚©”Û‚©
	Bid lastBid; // �‡ˆÓBid
	Bid max_bid; // Ž©•ª‚É‚Æ‚Á‚Ä‚Ì�Å‘åŒø—p’lBid
	Bid true_best_agreement_bid; // ‹­‹C‚Ì�ó‘Ô‚Ì‰ß‹Ž‚Ì�‡ˆÓBid‚Ì’†‚Å�Å‘åŒø—p’l‚Æ‚È‚éBid
	Bid false_best_agreement_bid; // Žã‹C‚Ì�ó‘Ô‚Ì‰ß‹Ž‚Ì�‡ˆÓBid‚Ì’†‚Å�Å‘åŒø—p’l‚Æ‚È‚éBid
	int true_failure_num; // ‹­‹C�ó‘Ô‚Å‚ÌŒð�ÂŽ¸”s�”
	int false_agreement_num; // Žã‹C�ó‘Ô‚Å‚ÌŒð�Â�¬Œ÷�”
	double false_agreement_average_utility_with_discount; // Žã‹C�ó‘Ô‚Å‚ÌƒfƒBƒXƒJƒEƒ“ƒg�ž‚Ý‚ÌŠl“¾Œø—p’l‚Ì•½‹Ï
	double partner_attack_num; // Œð�Â‘ŠŽè‚Ì‹­‹C‚Ì‰ñ�”
	double partner_guard_num; // Œð�Â‘ŠŽè‚ÌŽã‹C‚Ì‰ñ�”
	double last_time; // �ÅŒã‚ÌŒð�Â‚ð‚·‚éŽž��
	double last_bid_time; // ‘O‰ñ‚ÌŒð�Â�I—¹Žž��
	boolean tension_true_seal_flag; // ‹­‹C••ˆóƒtƒ‰ƒO
	double true_repeat_time; // ‹­‹C‚Ì�ÄŒ»”»’è—p‚ÌŽžŠÔ
	double false_repeat_time; // Žã‹C‚Ì�ÄŒ»”»’è—p‚ÌŽžŠÔ
	double false_agreement_best_utility; // ƒfƒBƒXƒJƒEƒ“ƒg�ž‚Ý‚ÌŠl“¾Œø—p’l‚Ì�Å‘å’l

	public MySessionData(Boolean tension, Boolean isAgreement, Bid lastBid,
			Bid max_bid, Bid true_best_agreement_bid,
			Bid false_best_agreement_bid, int true_failure_num,
			int false_agreement_num,
			double false_agreement_average_utility_with_discount,
			double partner_attack_num, double partner_guard_num,
			double last_time, double last_bid_time,
			boolean tension_true_seal_flag, double true_repeat_time,
			double false_repeat_time, double false_agreement_best_utility) {

		this.tension = tension;
		this.isAgreement = isAgreement;
		this.lastBid = lastBid;
		this.max_bid = max_bid;
		this.true_best_agreement_bid = true_best_agreement_bid;
		this.false_best_agreement_bid = false_best_agreement_bid;
		this.true_failure_num = true_failure_num;
		this.false_agreement_num = false_agreement_num;
		this.false_agreement_average_utility_with_discount = false_agreement_average_utility_with_discount;
		this.partner_attack_num = partner_attack_num;
		this.partner_guard_num = partner_guard_num;
		this.last_time = last_time;
		this.last_bid_time = last_bid_time;
		this.tension_true_seal_flag = tension_true_seal_flag;
		this.true_repeat_time = true_repeat_time;
		this.false_repeat_time = false_repeat_time;
		this.false_agreement_best_utility = false_agreement_best_utility;
	}
}