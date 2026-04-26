package agents.anac.y2014.AgentWhale;

import java.io.Serializable;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.NegotiationResult;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.DefaultAction;
import genius.core.actions.Offer;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.timeline.Timeline;

/**
 * This agent is an example of how to create an ANAC2013 agent which learns
 * during the tournament. This agent is a variant of the random agent.
 * 
 * @author M. Hendrikx
 */
public class WhaleAgent extends Agent {

	/** The minimum utility a bid should have to be accepted or offered. */
	private double MINIMUM_BID_UTILITY;
	private double MINIMUM_BID_UTILITY_OFFER = 1.0;

	private double offered_sum = 0.0;// ‘ŠŽè‚ªOffer‚µ‚Ä‚«‚½Bid‚ÌŒø—p’l‚Ì‘�˜a
	private Bid opponentBestBid;// ‘ŠŽè‚ªOffer‚µ‚Ä‚«‚½Bid‚Ì‚¤‚¿�Å‘å‚ÌŒø—p’l‚Ì‚à‚Ì
	private ArrayList<Bid> seedBidList;// ’T�õ‚ÌŽí‚Æ‚È‚éBidList
	private Bid opponentLastBid;// The opponent's last action.
	private Bid maxBid;// Bid with the highest possible utility.
	private boolean flag_offer_bestbid_opponent = false;
	private HashMap<Bid, Double> offeredBidMap;// ‘ŠŽè‚ÌOffer‚µ‚½Bid‚ð•Û‘¶‚µ‚Ä‚¨‚­HashMap
	private HashMap<String, Double> offeredMyBidMap;// Ž©•ª‚ÌOffer‚µ‚½Bid‚ð•Û‘¶‚µ‚Ä‚¨‚­HashMap
	private int countReSA = 0;
	private double lastTime;
	private boolean isFirstAction = false;
	private boolean isSecondAction = false;
	private ArrayList<Bid> offerBidLists = new ArrayList<Bid>();
	private int firstOfferIndex = 0;
	private int reOfferCount = 0;
	private boolean isFirstSession;
	private int actionType = 1;
	private int offered_count = 0;
	private double SAmax;

	private WhaleSessionData prevwData;

	public WhaleAgent() {
	}

	/**
	 * Initialize the target utility to MAX(rv, max). Where rv is the
	 * reservation value of the preference profile and max is the highest
	 * utility received on the current preference profile.
	 */
	@Override
	public void init() {
		// System.out.println("++ init here ++");

		Bid bid = utilitySpace.getDomain().getRandomBid(null);
		SAmax = utilitySpace.getUtilityWithDiscount(
				getBidSA(bid, 10000, 0.999, 0.90, 0), 0.0);

		// System.out.println("**SAmax "+SAmax);
		setName("WhaleAgent");
		// Serializable prev = this.loadSessionData();//
		// ‘O‰ñ‚ÌŠl“¾Œø—p’l
		// if (prev != null) {
		// double previousOutcome = (Double) prev;
		// MINIMUM_BID_UTILITY =
		// Math.max(Math.max(utilitySpace.getReservationValueUndiscounted(),previousOutcome),
		// 0.85);
		// } else {
		// MINIMUM_BID_UTILITY =
		// Math.max(utilitySpace.getReservationValueUndiscounted(), 0.85);
		// }
		//
		offered_sum = 0.0;
		offered_count = 0;
		MINIMUM_BID_UTILITY = 0.90;
		initPrevSessionData();
		// HashMap‚Ì�‰Šú‰»
		offeredBidMap = new HashMap<Bid, Double>();
		offeredMyBidMap = new HashMap<String, Double>();

		// predictType(true);

		// System.out.println("Minimum bid utility: " + MINIMUM_BID_UTILITY);
	}

	// Žc‚èŽžŠÔ‚ðŒ©‚Ä
	// Offer‚·‚é�@ƒrƒbƒh‚Ì�Å‘å’l‚ð‰º‚°‚é
	// Žc‚èŽžŠÔ‚ðŒ©‚Ä Accept‚·‚é�@
	public void calculateMyBid() {

	}

	private void calculateMinBidUtil() {
		double time = timeline.getTime();
		System.out.println("calculateMinBidUtil");
		System.out.println("++ time : " + time);
		double discount = utilitySpace.getDiscountFactor();
		if (discount < 1.0) {
			// Š„ˆøŒø—p‚ ‚è
			MINIMUM_BID_UTILITY = 1.0 * Math.pow(discount, time);
		} else {
			// Š„ˆøŒø—p‚È‚µ
			MINIMUM_BID_UTILITY = 1.0 - time;
		}

		// System.out.println("Minimum bid utility: " + MINIMUM_BID_UTILITY);
	}

	@Override
	public String getVersion() {
		return "1.0";
	}

	@Override
	public String getName() {
		return "WhaleAgent";
	}

	/**
	 * ƒZƒbƒVƒ‡ƒ“ƒf�[ƒ^‚ð•œŒ³‚·‚é
	 */
	public void initPrevSessionData() {

		try {

			// System.out.println("Session Start Data");
			// System.out.println("**sessionsTotal : "+sessionsTotal);
			// System.out.println("**sessionNr : "+sessionNr);

			Serializable prev = this.loadSessionData();

			if (prev != null) {

				// WhaleSessionDataAll wAll = (WhaleSessionDataAll) prev;
				WhaleSessionData wData = (WhaleSessionData) prev;
				prevwData = wData;
				HashMap<Bid, Double> tmpMap = new HashMap<Bid, Double>();
				for (int i = 0; i < prevwData.bidLists.size(); i++) {
					tmpMap.put(prevwData.bidLists.get(i),
							utilitySpace.getUtility(prevwData.bidLists.get(i)));
				}

				// Sort
				TreeMap<Bid, Double> treeMap = new TreeMap<Bid, Double>(
						new MapComparator(tmpMap));
				treeMap.putAll(tmpMap);

				prevwData.bidLists = new ArrayList<Bid>();
				for (Bid bid : treeMap.keySet()) {
					prevwData.bidLists.add(bid);
				}

				if (wData.getUtil >= 0.9) {
					// ‘O‰ñ‚ÌUtility‚ª�‚‚¢Žž =>
					// ‚»‚ÌBid‚ð‚·‚®‚ÉOffer‚·‚é‚æ‚¤‚É‚·‚é�iDiscount‚ª‚ ‚é‚Æ‚«�j
					reOfferCount = 1;
					isFirstAction = true;
					offerBidLists.add(wData.accesptedBid);

				} else if (wData.getUtil >= 0.8) {
					reOfferCount = 1;
					isFirstAction = true;
					offerBidLists.add(wData.accesptedBid);
				}
				// System.out.println(prev);
				isFirstSession = false;
			} else {
				reOfferCount = 0;
				isFirstSession = true;
				offerBidLists = new ArrayList<Bid>();// �‰Šú‰»
			}

			actionType = predictType(false);// Actionƒpƒ^�[ƒ“‚ð•Ï�X‚·‚é
			// if(isFirstAction == false && actionType == 2){
			// isFirstAction = true;
			// }

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * Set the target utility for the next match on the same preference profile.
	 * If the received utility is higher than the current target, save the
	 * received utility as the new target utility.
	 */
	@Override
	public void endSession(NegotiationResult result) {
		// System.out.println("endSession");
		if (result.getMyDiscountedUtility() > MINIMUM_BID_UTILITY) {
			// System.out.println("saveSession");
			saveSessionData(new Double(result.getMyDiscountedUtility()));
			// saveSessionData(offeredBidMap);
		}
		// System.out.println(result);
		if (!result.isAgreement()) {
			// �‡ˆÓ‚Å‚«‚È‚©‚Á‚½Žž

			// ŽŸ‰ñ‚©‚ç‘Ã‹¦‚·‚é‚æ‚¤‚É‚·‚é
		}

		ArrayList<Bid> bidLists;
		double sumUtil = 0.0;
		// HashMap<Bid, Double> bestMaps = getBestBids(3);
		if (isFirstSession) {
			bidLists = new ArrayList<Bid>();
		} else {
			sumUtil = prevwData.sumUtil + result.getMyDiscountedUtility();// recalculate
			bidLists = new ArrayList<Bid>(prevwData.bidLists);

		}

		bidLists.add(result.getLastBid());

		WhaleSessionData sessionData = new WhaleSessionData(result.getLastBid(),
				getBestBidOpponent(), result.getMyDiscountedUtility(),
				result.isAgreement(), bidLists, sumUtil, actionType);

		saveSessionData(sessionData);
		// System.out.println("**save");

		// TreeMap<Bid, Double> treeMap = new TreeMap<Bid, Double>(new
		// IntegerMapComparator(bestMaps));
		// treeMap.putAll(bestMaps);
		// System.out.println(treeMap.values());

	}

	/**
	 * Offer‚·‚é‘O‚ÉHashMap‚É•Û‘¶‚·‚éƒ�ƒ\ƒbƒh
	 * 
	 * @param bid
	 * @return
	 */
	public Action preOffer(Bid bid) {
		// Offer‚·‚é‘O‚ÉHashMap‚ÉŽ©•ª‚ªOffer‚µ‚½Bid‚ð•Û‘¶‚µ‚Ä‚¨‚­
		offeredMyBidMap.put(bid.toString(),
				utilitySpace.getUtilityWithDiscount(bid, timeline.getTime()));
		return new Offer(getAgentID(), bid);
	}

	/**
	 * �¡‚Ü‚ÅŽ©•ª‚ªOffer‚µ‚Ä‚«‚½Bid‚©‚Ç‚¤‚©
	 * 
	 * @param bid
	 * @return
	 */
	public Boolean isCollideBid(Bid bid) {
		if (offeredMyBidMap.containsKey(bid.toString())) {
			return true;
		} else {
			return false;
		}
	}

	public Bid getBestSeedBid() {

		double best_util = 0.0;
		Bid bestBid = utilitySpace.getDomain().getRandomBid(null);
		// ‰ß‹Ž‚ÌBid‚æ‚èˆê’è‚Ì’l‚æ‚è‘å‚«‚¢Bid‚ðƒ‰ƒ“ƒ_ƒ€‚ÅSeed‚Æ‚µ‚Ä�Ì—p
		for (Bid bid : offeredBidMap.keySet()) {
			double recent_util = utilitySpace.getUtilityWithDiscount(bid,
					timeline.getTime());
			if (recent_util > best_util) {
				bestBid = bid;
				best_util = recent_util;
			}
		}
		return bestBid;
	}

	public Bid getMinBidMaps(HashMap<Bid, Double> maps) {
		double min_util = 1.0;
		Bid minBid = null;
		for (Bid bid : maps.keySet()) {
			double util = maps.get(bid);
			if (min_util >= util) {
				min_util = util;
				minBid = bid;
			}
		}
		return minBid;
	}

	public HashMap<Bid, Double> getBestBids(int count) {
		HashMap<Bid, Double> maps = new HashMap<Bid, Double>();

		try {

			// Sort
			TreeMap<Bid, Double> treeMap = new TreeMap<Bid, Double>(
					new MapComparator(offeredBidMap));
			treeMap.putAll(offeredBidMap);

			int i = 0;
			for (Bid bid : treeMap.keySet()) {
				if (i < count) {
					double recent_util;
					recent_util = utilitySpace.getUtility(bid);

					maps.put(bid, recent_util);
				}
				i++;
			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return maps;

	}

	/**
	 * ‘ŠŽè‚ÌOffer‚µ‚Ä‚«‚½Bid‚Ì‚¤‚¿�Å�‚‚ÌŒø—p’l‚ðŽ�‚Â‚à‚Ì‚ðŽæ“¾‚·‚é
	 * 
	 * @return
	 */
	public Bid getBestBidOpponent() {
		double best_util = 0.0;
		Bid bestBid = null;
		// ‰ß‹Ž‚ÌBid‚æ‚èˆê’è‚Ì’l‚æ‚è‘å‚«‚¢Bid‚ðƒ‰ƒ“ƒ_ƒ€‚ÅSeed‚Æ‚µ‚Ä�Ì—p
		for (Bid bid : offeredBidMap.keySet()) {
			double recent_util = utilitySpace.getUtilityWithDiscount(bid,
					timeline.getTime());
			if (recent_util > best_util) {
				bestBid = bid;
				best_util = recent_util;
			}
		}
		return bestBid;
	}

	/**
	 * ‘ŠŽè‚Ì’ñˆÄ‚µ‚Ä‚«‚½Bid‚©‚çƒxƒXƒg‚ÈBid‚ðŽæ“¾‚·‚é
	 * 
	 * @return
	 */
	public Action getBestBidByOppo(int type) {

		Bid bestBid = getBestBidOpponent();
		double best_util = utilitySpace.getUtilityWithDiscount(bestBid,
				timeline.getTime());
		if (flag_offer_bestbid_opponent == true || type == 0) {
			// ‘ŠŽè‚ÌBestBid‚ðOffer‚µ‚½‚Ì‚É‹‘”Û‚³‚ê‚½Žž
			// System.out.println("**newoffer");
			// ‘ŠŽè‚ÌBestBidŽü•Ó‚ð’T�õ
			return preOffer(getNeighborBid(bestBid));
			// return getBidLowSA(bestBid,100, 0.9,best_util,0);
		}

		if (best_util >= utilitySpace
				.getReservationValueWithDiscount(timeline.getTime())
				&& type == 1) {
			// ‘ŠŽè‚ÌBestBid‚ðOffer‚·‚é
			flag_offer_bestbid_opponent = true;

			return preOffer(bestBid);
		} else {
			Bid bid = utilitySpace.getDomain().getRandomBid(null);// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬
			return preOffer(getNeighborBid(opponentLastBid));
			// return getBidSA(bid);
		}

		// Random rnd = new Random();
		// if(rnd.nextInt(20000) % 5 == 0 ){
		// //5‰ñ‚É1‰ñ‚ÍŽ©—R‚ÉOffer‚·‚é
		// Bid bid = utilitySpace.getDomain().getRandomBid();//
		// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬
		// return getBidSA(bid);
		// }

		// return new Offer(bestBid);
	}

	/**
	 * i‰ñ‚©‚P‰ñtrue‚Æ‚È‚éƒ�ƒ\ƒbƒh
	 * 
	 * @param i
	 * @return
	 */
	public boolean getRandomTrue(int i) {
		Random rnd = new Random();
		int random_int = rnd.nextInt(10000);
		return random_int % i == 0;

	}

	/**
	 * offeredBidMap‚æ‚èŒø—p’l‚Ì�‚‚¢BidŒS‚ðŽæ“¾‚·‚é
	 */
	public Bid getSeedBid() {
		seedBidList = new ArrayList<Bid>();

		// ‰ß‹Ž‚ÌBid‚æ‚èˆê’è‚Ì’l‚æ‚è‘å‚«‚¢Bid‚ðƒ‰ƒ“ƒ_ƒ€‚ÅSeed‚Æ‚µ‚Ä�Ì—p
		for (Bid bid : offeredBidMap.keySet()) {
			if (offeredBidMap.get(bid) > 0) {// TODO�@‚¢‚­‚ÂˆÈ�ã‚Ìutil‚ðSeed‚Æ‚µ‚Ä�Ì—p‚·‚é‚©
				seedBidList.add(bid);
			}
		}

		if (getRandomTrue(5)) {

			return utilitySpace.getDomain().getRandomBid(null);// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬
		} else {
			if (!seedBidList.isEmpty() && seedBidList.size() > 3) {
				Random rnd = new Random();
				int seed_id = rnd.nextInt(seedBidList.size() - 1);// •Ï�X‚·‚é
				Bid seedBid = seedBidList.get(seed_id);

				return seedBid;

			} else {
				// ‚Ü‚¾‘ŠŽè‚©‚ç‚ÌBid‚ª‚È‚¢‚Æ‚«
				return utilitySpace.getDomain().getRandomBid(null);// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬

			}
		}

	}

	/**
	 * Retrieve the bid from the opponent's last action.
	 */
	@Override
	public void ReceiveMessage(Action opponentAction) {

		opponentLastBid = DefaultAction.getBidFromAction(opponentAction);// ‘ŠŽè‚ÌlastBid

		if (opponentLastBid != null) {

			// ‘ŠŽè‚ÌOffer‚µ‚Ä‚«‚½Bid‚ðŽæ“¾‚µ�A•Û‘¶‚·‚é
			double time = timeline.getTime();// Œo‰ßŽžŠÔ‚ðŽæ“¾
			double offeredUtility = utilitySpace
					.getUtilityWithDiscount(opponentLastBid, time);// discount‚³‚ê‚½‚à‚Ì‚ðŽæ“¾
			offeredBidMap.put(opponentLastBid, offeredUtility);// HashMap‚É•Û‘¶‚·‚é
			offered_count += 1;
			offered_sum += offeredUtility;
		}

		// System.out.println("+ oopo : "+opponentLastBid.getValues());
		// System.out.println("+ReservationValue :
		// "+utilitySpace.getReservationValue());
		// System.out.println("++ opponet : "+getUtility(opponentLastBid));
		// TODO
		// lastBid‚ÌŠˆ—p�@‚µ‚«‚¢’l‚ð‰º‚°‚½‚è‚·‚é
		// System.out.println("Time : "+timeline.getTime());

		// MINIMUM_BID_UTILITY_OFFER = sigmoid(timeline.getTime());
		// MINIMUM_BID_UTILITY = sigmoid(timeline.getTime());

	}

	/**
	 * ‘ŠŽè‚Ì�s“®ƒpƒ^�[ƒ“‚©‚çŽ©•ª‚Ì�s“®ƒpƒ^�[ƒ“‚ðŒˆ‚ß‚é
	 * 
	 * @return
	 */
	public int predictType(boolean isPre) {
		if (isPre == false) {
			// System.out.println(offered_count);
			// System.out.println(offerBidLists.size());
			// System.out.println("**offeredBidMap.size() =
			// "+offeredBidMap.size());
			// System.out.println("**offered_count = "+offered_count);
			if (offered_count > 10
					&& offered_count > 2 * offeredBidMap.size()) {
				// System.out.println("** 2 count");
				return 2;
			}
			//
		}

		if (isFirstSession) {
			// ‰ß‹Ž‚ÌBid‚ª‚È‚¢�ê�‡
			// System.out.println("** 1 first");
			return 1;
		} else {
			// ‘O‰ñ‚Ì�î•ñ‚ð—˜—p‚·‚é

			if (prevwData.isAgreement == false) {
				// System.out.println("** 2 not agreement");
				return 2;
			}
			if (prevwData.getUtil < 0.7 * SAmax) {
				// System.out.println("** 2 low util");
				return 2;
			}

			double sumUtil = prevwData.sumUtil;// ‚±‚ÌƒZƒbƒVƒ‡ƒ“‚Ü‚Å‚Ì�‡Œv
			double avgUtil = sumUtil / (sessionNr + 1.0);
			// System.out.println("**sessionNr "+sessionNr);
			// •½‹Ï‚ª�‚‚¯‚ê‚Î
			if (avgUtil > 0.8 * SAmax) {
				// System.out.println("** 1 avg");
				return 1;
			}
			// System.out.println("** prev end"+prevwData.actionType);
			return prevwData.actionType;
		}
		// System.out.println("** 2 end");
		// return 2;
	}

	/**
	 * ‹­‹C‚Ì�s“®‚ð‚·‚éƒ�ƒ\ƒbƒh
	 * 
	 * @return
	 */
	public Action actionType1() {
		// System.out.println("**actionType1");
		try {
			// ŽžŠÔ‚ð�l—¶‚µ‚Ämin
			// bid‚ð•Ï‰»‚³‚¹‚é
			// calculateMinBidUtil();
			// System.out.println("hashmap_count : "+offeredBidMap.size());
			// System.out.println("hashmap_util : "+offered_sum);
			// System.out.println("hashmap_avg :
			// "+offered_sum/offeredBidMap.size());
			if (isFirstAction) {

				for (int i = firstOfferIndex; i < offerBidLists.size(); i++) {
					firstOfferIndex++;
					if (firstOfferIndex == offerBidLists.size() - 1) {
						isFirstAction = false;
						reOfferCount = 2;
					}
					// System.out.println("**new first offer");
					isSecondAction = true;
					return preOffer(offerBidLists.get(i));
				}
			}

			if (opponentLastBid != null && utilitySpace
					.getUtility(opponentLastBid) >= MINIMUM_BID_UTILITY) {
				return new Accept(getAgentID(), opponentLastBid);

			} else if (opponentLastBid != null && utilitySpace.getUtility(
					opponentLastBid) >= MINIMUM_BID_UTILITY - 0.05) {
				// Œë�·‚ª0.1‚ÌŽž
				if (getRandomTrue(3)) {
					return new Accept(getAgentID(), opponentLastBid);
				}

			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 2) {

				if (prevwData.acceptTime / 1.3 <= timeline.getTime()) {

					reOfferCount = 3;
					return preOffer(offerBidLists.get(0));

				}
			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 3) {
				if (prevwData.acceptTime / 1.2 <= timeline.getTime()) {
					reOfferCount = 4;
					return preOffer(offerBidLists.get(0));
				}
			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 4) {
				if (prevwData.acceptTime / 1.1 <= timeline.getTime()) {
					reOfferCount = 5;
					return preOffer(offerBidLists.get(0));
				}
			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 5) {
				if (prevwData.acceptTime <= timeline.getTime()) {
					reOfferCount = 6;
					return preOffer(offerBidLists.get(0));
				}
			}
			if (timeline.getTime() > 0.995) {
				double util = getUtility(opponentLastBid);
				if (util > utilitySpace
						.getReservationValueWithDiscount(timeline)) { // TODO:Resevation
																		// Value‚Æ”ä‚×‚é•K—v‚ ‚è
					return new Accept(getAgentID(), opponentLastBid);
				}
				// Žc‚èŽžŠÔ‚ª�­‚È‚¢Žž
				// return
				// getBestBidByOppo(1);//‘Ã‹¦ƒ‚�[ƒh‚ÌŽž�H

			}
			if (timeline.getTime() > 0.97) {

				// Žc‚èŽžŠÔ‚ª�­‚È‚¢Žž
				// return getBestBidByOppo(1);

			}
			if (timeline.getTime() > 0.91) {

				double util = utilitySpace.getUtilityWithDiscount(
						getBestBidOpponent(), timeline.getTime());
				if (util < MINIMUM_BID_UTILITY) {
					// MINIMUM_BID_UTILITY =
					// util*1.05;//TODO:‚±‚±‚ðƒRƒ�ƒ“ƒgƒAƒEƒg‚·‚é‚Æ“¦‚°‚é“®‚«‚ð‚·‚é‚Ì‚Å‚¢‚¢‚©‚à�H
				}
				// Žc‚èŽžŠÔ‚ª�­‚È‚¢Žž
				// return getBestBidByOppo(0);

			}

			if (timeline.getTime() > 0.8) {
				MINIMUM_BID_UTILITY = 0.99 * MINIMUM_BID_UTILITY * SAmax;

				// }else if(timeline.getTime() > 0.5){
				// Bid bid =
				// getSeedBid();//’T�õ‚ÌSeedBid‚ðŽæ“¾‚·‚é
				// return getBidLowSA(bid,5000, 0.98,0.8,0);//5000
			}

			if (timeline.getTime() > 0.1) {
				// Bid bid = getSeedBid();

				Bid bid = getNeighborBid(getBestSeedBid());
				double util = getUtility(bid);
				if (util >= MINIMUM_BID_UTILITY) {
					// System.out.println("** neighbor");
					return preOffer(bid);
				}
				// return getBidHighClimb(bid, 50, 0.90, 0);
				// return getBidLowSA(bid, 10000,0.999,0.90,0);
				// return getBidLowSA(bid, 1000,0.8,0.90,0);

				// Bid bid = getBestBidOpponent();
				// return getBidHighClimb(bid, 10000,0.98,0.90,0);
			}

			// System.out.println("t1 : "+Timeline.Type.Time);
			// System.out.println("t2 : "+timeline.getType());
			if (timeline.getType().equals(Timeline.Type.Time)) {
				// sleep(0.0005);//just for fun
			}
			// return getRandomBid(MINIMUM_BID_UTILITY);
			// return getMyBid();

			// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬‚·‚é
			// Bid bid = utilitySpace.getDomain().getRandomBid();//
			// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// return getBidSA(bid);

		// Bid bid =
		// getBestSeedBid();//’T�õ‚ÌSeedBid‚ðŽæ“¾‚·‚é
		Bid bid = getSeedBid();
		// return getBidLowSA(bid, 10000,0.98,0.90,0);
		// System.out.println("** SA last");
		return getBidLowSA(bid, 10000, 0.98, MINIMUM_BID_UTILITY * SAmax, 0);
	}

	/**
	 * Žã‹C‚Ì�s“®‚ð‚·‚éƒ�ƒ\ƒbƒh
	 * 
	 * @return
	 */
	public Action actionType2() {

		// System.out.println("**actionType2");
		try {
			// ŽžŠÔ‚ð�l—¶‚µ‚Ämin
			// bid‚ð•Ï‰»‚³‚¹‚é
			// calculateMinBidUtil();
			// System.out.println("hashmap_count : "+offeredBidMap.size());
			// System.out.println("hashmap_util : "+offered_sum);
			// System.out.println("hashmap_avg :
			// "+offered_sum/offeredBidMap.size());
			if (isFirstAction) {

				for (int i = firstOfferIndex; i < offerBidLists.size(); i++) {
					firstOfferIndex++;
					if (firstOfferIndex == offerBidLists.size() - 1) {
						isFirstAction = false;
						reOfferCount = 2;
					}

					return preOffer(offerBidLists.get(i));
				}
			}

			if (opponentLastBid != null && utilitySpace
					.getUtility(opponentLastBid) >= MINIMUM_BID_UTILITY) {
				return new Accept(getAgentID(), opponentLastBid);

			} else if (opponentLastBid != null && utilitySpace.getUtility(
					opponentLastBid) >= MINIMUM_BID_UTILITY - 0.05) {
				// Œë�·‚ª0.1‚ÌŽž
				if (getRandomTrue(3)) {
					return new Accept(getAgentID(), opponentLastBid);
				}

			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 2) {

				if (prevwData.acceptTime / 1.3 <= timeline.getTime()) {

					reOfferCount = 3;
					return preOffer(offerBidLists.get(0));

				}
			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 3) {
				if (prevwData.acceptTime / 1.2 <= timeline.getTime()) {
					reOfferCount = 4;
					return preOffer(offerBidLists.get(0));
				}
			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 4) {
				if (prevwData.acceptTime / 1.1 <= timeline.getTime()) {
					reOfferCount = 5;
					return preOffer(offerBidLists.get(0));
				}
			}
			if (!isFirstSession && !isFirstAction && reOfferCount == 5) {
				if (prevwData.acceptTime <= timeline.getTime()) {
					reOfferCount = 6;
					return preOffer(offerBidLists.get(0));
				}
			}
			if (timeline.getTime() > 0.99) {
				double util = getUtility(opponentLastBid);
				if (util > utilitySpace.getReservationValue()) { // TODO:Resevation
																	// Value‚Æ”ä‚×‚é•K—v‚ ‚è
					return new Accept(getAgentID(), opponentLastBid);
				}
				// Žc‚èŽžŠÔ‚ª�­‚È‚¢Žž
				return getBestBidByOppo(1);// ‘Ã‹¦ƒ‚�[ƒh‚ÌŽž�H

			}
			if (timeline.getTime() > 0.97) {

				// Žc‚èŽžŠÔ‚ª�­‚È‚¢Žž
				return getBestBidByOppo(1);

			}
			if (timeline.getTime() > 0.91) {

				double util = utilitySpace.getUtilityWithDiscount(
						getBestBidOpponent(), timeline.getTime());
				if (util < MINIMUM_BID_UTILITY) {
					MINIMUM_BID_UTILITY = util * 0.99 * SAmax;
				}
				// Žc‚èŽžŠÔ‚ª�­‚È‚¢Žž
				return getBestBidByOppo(0);

			}

			if (timeline.getTime() > 0.8) {
				MINIMUM_BID_UTILITY = 0.85 * SAmax;

			}

			if (timeline.getTime() > 0.5) {
				// Bid bid =
				// getSeedBid();//’T�õ‚ÌSeedBid‚ðŽæ“¾‚·‚é
				//
				// return getBidHighClimb(bid, 5, 0.80, 0);

				Bid bid = getNeighborBid(getBestSeedBid());
				double util = getUtility(bid);
				if (util >= MINIMUM_BID_UTILITY) {
					return preOffer(bid);
				}
				// return getBidLowSA(bid,5000, 0.98,0.8,0);//5000
			}

			if (timeline.getTime() > 0.5) {
				MINIMUM_BID_UTILITY = 0.70 * SAmax;

			}
			if (timeline.getTime() > 0.1) {
				Bid bid = getNeighborBid(getBestSeedBid());
				double util = getUtility(bid);
				if (util >= 0.80 * SAmax) {
					return preOffer(bid);
				}

				// Bid bid = getSeedBid();
				// return getBidHighClimb(bid, 5, 0.80, 0);
				// return getBidLowSA(bid, 10000,0.98,0.90,0);
				// Bid bid = getBestBidOpponent();
				// return getBidHighClimb(bid, 10000,0.98,0.90,0);
			}

			// System.out.println("t1 : "+Timeline.Type.Time);
			// System.out.println("t2 : "+timeline.getType());
			// if (timeline.getType().equals(Timeline.Type.Time)) {
			// sleep(0.0005);//just for fun
			// }
			// return getRandomBid(MINIMUM_BID_UTILITY);
			// return getMyBid();

			// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬‚·‚é
			// Bid bid = utilitySpace.getDomain().getRandomBid();//
			// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// return getBidSA(bid);

		Bid bid = getBestSeedBid();// ’T�õ‚ÌSeedBid‚ðŽæ“¾‚·‚é

		return getBidLowSA(bid, 10000, 0.98, 0.85 * SAmax, 0);

	}

	/**
	 * Accept if the utility of the opponent's is higher than the target
	 * utility; else return a random bid with a utility at least equal to the
	 * target utility.
	 */
	@Override
	public Action chooseAction() {// Ž©•ª‚ª‰½‚ð‚·‚é‚©
		// System.out.println("chooseAction");

		actionType = predictType(false);// Actionƒpƒ^�[ƒ“‚ð•Ï�X‚·‚é

		if (actionType == 1) {
			return actionType1();// ‹­‹C‚Ì�s“®ƒpƒ^�[ƒ“
		} else {
			return actionType2();// Žã‹C‚Ì�s“®ƒpƒ^�[ƒ“
		}

	}

	public double sigmoid(double x) {
		// return (Math.exp(3*x)+Math.exp(-5*x))/(Math.exp(3*x)+Math.exp(-x));
		// return 1.0/(Math.pow(x, 30)+Math.exp(Math.pow(x, 100)));
		// return -1.0/(1+Math.exp(2/x))+1;
		// return -4.0/(10+Math.exp(2.0/x))+1;

		//
		// return
		// (-1)/(1+Math.exp(-50*(x-1)))+1;//0.9•t‹ß‚©‚ç‹}�~‰º�@
		return (-1) / (1 + Math.exp(-20 * (x - 1))) + 1;// 0.9•t‹ß‚©‚ç‹}�~‰º�@
		// return (Math.pow(x,
		// 16)+Math.exp(10*x-15))/(Math.exp(-x)+Math.exp(x))+1;//x=0.8•t‹ß‚©‚ç‰º‚ª‚è�Ay=0.8‚­‚ç‚¢‚Ü‚Å‰º‚ª‚é
		// return
		// Math.sin(x)*Math.cos(x)+Math.exp(-x)-0.05;//�Å�‰ŠÉ‚­�A�ÅŒãŒµ‚µ‚­Œ^
		// double result=
		// Math.sin(x)*Math.cos(x)+Math.exp(-10*x+Math.log(10*x+0.02))+0.5;//‚Q‚Â‚Ìƒs�[ƒN‚ðŽ�‚ÂƒOƒ‰ƒt

		// if(result < 0.8){
		// return 0.8;
		// }else{
		// return result;
		// }
		// y=sin({x})|_cdot_cos({x})+exp({-10x+log({10x+0.02})})+0.5

		// return (-1.0)/(1.0+Math.exp(-30.0*x+31))+1;
	}

	/**
	 * Žü•ÓBid‚ð’T�õ‚·‚é
	 * 
	 * @param bid
	 * @return
	 */
	public Bid getNeighborBid(Bid bid) {
		try {
			Random rnd = new Random();
			int step = 3;
			for (int i = 0; i < bid.getIssues().size(); i++) {
				double bf = utilitySpace.getUtility(bid);
				Bid bidChange = new Bid(bid);

				int flag = rnd.nextBoolean() ? 1 : -1;// 1=> plus , -1 => minus
				// int change_id = i;
				int change_id = rnd.nextInt(bid.getIssues().size());// •Ï�X‚·‚é
				IssueInteger issueInteger = (IssueInteger) bid.getIssues()
						.get(change_id);
				int issueId = issueInteger.getNumber();
				ValueInteger issueValue = (ValueInteger) bid.getValue(issueId);
				int issueValueInt = Integer.valueOf(issueValue.toString())
						.intValue();
				int max = issueInteger.getUpperBound();// �Å‘å’l
				int min = issueInteger.getLowerBound();// �Å�¬’l

				if (issueValueInt >= min && issueValueInt <= max) {

					if (issueValueInt + step > max) {
						// ‘«‚·‚Æ–â‘è‚ ‚è
						flag = -1;
					} else if (issueValueInt - step < min) {
						// ˆø‚­‚Æ–â‘è‚ ‚è
						flag = 1;
					}

				}

				Value valueInteger = new ValueInteger(
						issueValueInt + flag * step);// change
														// value

				bidChange = bidChange.putValue(issueId, valueInteger);// change

				double af = utilitySpace.getUtility(bidChange);

				double bf_cost = 1.0 - bf;
				double af_cost = 1.0 - af;

				if (af_cost < bf_cost) {
					bid = new Bid(bidChange);
				}

			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return (bid);

	}

	/**
	 * ‰·“x’á‚ß‚ÌSA
	 * 
	 * @param bid
	 * @param T
	 * @return
	 */
	public Action getBidLowSA(Bid bid, double T, double cool, double bias,
			int loop_count) {

		// �Ä‚«“Ý‚µ–@
		// Bid bid = utilitySpace.getDomain().getRandomBid();//
		// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬
		// System.out.println(bid.getValues());
		int count = 0;
		try {
			// T = 10000;

			int step = 1;// •Ï�X‚·‚é•�
			// double cool = 0.6;// ‰·“x‚ð‰º‚°‚é•�
			Random rnd = new Random();

			while (T > 0.0001) {
				boolean ok_flag = false;
				int ok_count = 0;
				count++;
				double bf = utilitySpace.getUtility(bid);
				Bid bidChange = new Bid(bid);

				// bid change

				int change_id = rnd.nextInt(bid.getIssues().size());// •Ï�X‚·‚é
																	// IssueId‚ÌƒCƒ“ƒfƒbƒNƒX‚ðŒˆ’è
				IssueInteger issueInteger = (IssueInteger) bid.getIssues()
						.get(change_id);
				int issueId = issueInteger.getNumber();
				ValueInteger issueValue = (ValueInteger) bid.getValue(issueId);
				int issueValueInt = Integer.valueOf(issueValue.toString())
						.intValue();
				int max = issueInteger.getUpperBound();// �Å‘å’l
				int min = issueInteger.getLowerBound();// �Å�¬’l

				int flag = rnd.nextBoolean() ? 1 : -1;// 1=> plus , -1 => minus
				if (issueValueInt >= min && issueValueInt <= max) {

					if (issueValueInt + step > max) {
						// ‘«‚·‚Æ–â‘è‚ ‚è
						flag = -1;
					} else if (issueValueInt - step < min) {
						// ˆø‚­‚Æ–â‘è‚ ‚è
						flag = 1;
					}

				}

				Value valueInteger = new ValueInteger(
						issueValueInt + flag * step);// change
														// value
				bidChange = bidChange.putValue(issueId, valueInteger);// change

				double af = utilitySpace.getUtility(bidChange);

				double bf_cost = 1.0 - bf;
				double af_cost = 1.0 - af;
				// System.out.println("af : "+af);
				double p = Math.pow(Math.E, -Math.abs(af_cost - bf_cost) / T);
				// System.out.println(" p = "+p);
				// System.out.println(" rnd = "+rnd.nextDouble());

				// System.out.println(rnd.nextDouble() < p);
				if ((af_cost < bf_cost || rnd.nextDouble() < p)
						&& af <= MINIMUM_BID_UTILITY_OFFER) {
					bid = new Bid(bidChange);
					if (af >= bias) {// TODO:
										// ‚±‚ê‚Å‚¢‚¢‚Ì‚©�H
						ok_flag = true;

						if (ok_count >= rnd.nextInt(3)) {
							ok_count = 0;
							ok_flag = false;
							// break;
						}

					}
					if (ok_flag) {
						ok_count++;
					}
				}

				NumberFormat format = NumberFormat.getInstance();
				format.setMaximumFractionDigits(1);

				T = T * cool;

			}

			if (utilitySpace.getUtility(bid) < bias) {
				// System.out.println("**reSA");

				countReSA = loop_count + 1;
				double t_cool = (0.999 - cool) / 20.0;
				return getBidLowSA(bid, step, cool + t_cool, bias,
						loop_count + 1);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// System.out.println(count);
		if (loop_count > 0) {
			// System.out.println("loop_count: "+loop_count);
		}

		return preOffer(bid);
	}

	public Bid getBidSA(Bid bid, double T, double cool, double bias,
			int loop_count) {

		// �Ä‚«“Ý‚µ–@
		// Bid bid = utilitySpace.getDomain().getRandomBid();//
		// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬
		// System.out.println(bid.getValues());
		int count = 0;
		try {
			// T = 10000;

			int step = 1;// •Ï�X‚·‚é•�
			// double cool = 0.6;// ‰·“x‚ð‰º‚°‚é•�
			Random rnd = new Random();

			while (T > 0.0001) {
				boolean ok_flag = false;
				int ok_count = 0;
				count++;
				double bf = utilitySpace.getUtility(bid);
				Bid bidChange = new Bid(bid);

				// bid change

				int change_id = rnd.nextInt(bid.getIssues().size());// •Ï�X‚·‚é
																	// IssueId‚ÌƒCƒ“ƒfƒbƒNƒX‚ðŒˆ’è
				IssueInteger issueInteger = (IssueInteger) bid.getIssues()
						.get(change_id);
				int issueId = issueInteger.getNumber();
				ValueInteger issueValue = (ValueInteger) bid.getValue(issueId);
				int issueValueInt = Integer.valueOf(issueValue.toString())
						.intValue();
				int max = issueInteger.getUpperBound();// �Å‘å’l
				int min = issueInteger.getLowerBound();// �Å�¬’l

				int flag = rnd.nextBoolean() ? 1 : -1;// 1=> plus , -1 => minus
				if (issueValueInt >= min && issueValueInt <= max) {

					if (issueValueInt + step > max) {
						// ‘«‚·‚Æ–â‘è‚ ‚è
						flag = -1;
					} else if (issueValueInt - step < min) {
						// ˆø‚­‚Æ–â‘è‚ ‚è
						flag = 1;
					}

				}

				Value valueInteger = new ValueInteger(
						issueValueInt + flag * step);// change
														// value
				bidChange = bidChange.putValue(issueId, valueInteger);// change

				double af = utilitySpace.getUtility(bidChange);

				double bf_cost = 1.0 - bf;
				double af_cost = 1.0 - af;
				// System.out.println("af : "+af);
				double p = Math.pow(Math.E, -Math.abs(af_cost - bf_cost) / T);
				// System.out.println(" p = "+p);
				// System.out.println(" rnd = "+rnd.nextDouble());

				// System.out.println(rnd.nextDouble() < p);
				if ((af_cost < bf_cost || rnd.nextDouble() < p)
						&& af <= MINIMUM_BID_UTILITY_OFFER) {
					bid = new Bid(bidChange);
					if (af >= bias) {// TODO:
										// ‚±‚ê‚Å‚¢‚¢‚Ì‚©�H
						ok_flag = true;

						if (ok_count >= rnd.nextInt(3)) {
							ok_count = 0;
							ok_flag = false;
							// break;
						}

					}
					if (ok_flag) {
						ok_count++;
					}
				}

				NumberFormat format = NumberFormat.getInstance();
				format.setMaximumFractionDigits(1);

				T = T * cool;

			}

			if (utilitySpace.getUtility(bid) < bias) {
				// System.out.println("**reSA");

				countReSA = loop_count + 1;
				if (loop_count >= 10) {
					return bid;
				}
				double t_cool = (0.999 - cool) / 20.0;
				return getBidSA(bid, step, cool + t_cool, bias, loop_count + 1);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// System.out.println(count);
		if (loop_count > 0) {
			// System.out.println("loop_count: "+loop_count);
		}

		return (bid);
	}

	/**
	 * ‰ü—Ç”Å�Ä‚«‚È‚Ü‚µ�•ŽR“o‚è–@
	 * 
	 * @param bid
	 * @return
	 */
	public Action getBidHighClimb(Bid bid, int loop, double bias,
			int loop_count) {
		try {
			// T = 10000;

			int count = 0;
			int step = 1;// •Ï�X‚·‚é•�
			// double cool = 0.6;// ‰·“x‚ð‰º‚°‚é•�
			Random rnd = new Random();

			while (loop > count) {
				boolean ok_flag = false;
				int ok_count = 0;
				count++;
				double bf = utilitySpace.getUtility(bid);
				Bid bidChange = new Bid(bid);

				// bid change

				int change_id = rnd.nextInt(bid.getIssues().size());// •Ï�X‚·‚é
																	// IssueId‚ÌƒCƒ“ƒfƒbƒNƒX‚ðŒˆ’è
				IssueInteger issueInteger = (IssueInteger) bid.getIssues()
						.get(change_id);
				int issueId = issueInteger.getNumber();
				ValueInteger issueValue = (ValueInteger) bid.getValue(issueId);
				int issueValueInt = Integer.valueOf(issueValue.toString())
						.intValue();
				int max = issueInteger.getUpperBound();// �Å‘å’l
				int min = issueInteger.getLowerBound();// �Å�¬’l

				int flag = rnd.nextBoolean() ? 1 : -1;// 1=> plus , -1 => minus
				if (issueValueInt >= min && issueValueInt <= max) {

					if (issueValueInt + step > max) {
						// ‘«‚·‚Æ–â‘è‚ ‚è
						flag = -1;
					} else if (issueValueInt - step < min) {
						// ˆø‚­‚Æ–â‘è‚ ‚è
						flag = 1;
					}

				}

				Value valueInteger = new ValueInteger(
						issueValueInt + flag * step);// change
														// value
				bidChange = bidChange.putValue(issueId, valueInteger);// change

				double af = utilitySpace.getUtility(bidChange);

				double bf_cost = 1.0 - bf;
				double af_cost = 1.0 - af;
				// System.out.println("af : "+af);
				// double p = Math.pow(Math.E, -Math.abs(af_cost - bf_cost) /
				// T);
				// System.out.println(" p = "+p);
				// System.out.println(" rnd = "+rnd.nextDouble());
				if ((af_cost < bf_cost) && af <= MINIMUM_BID_UTILITY_OFFER) {
					// count++;
					bid = new Bid(bidChange);
					if (af >= bias) {// TODO:
										// ‚±‚ê‚Å‚¢‚¢‚Ì‚©�H
						ok_flag = true;

						if (ok_count >= rnd.nextInt(3)) {
							ok_count = 0;
							ok_flag = false;
							// break;
						}

					}
					if (ok_flag) {
						ok_count++;
					}
				}

				NumberFormat format = NumberFormat.getInstance();
				format.setMaximumFractionDigits(1);

				// T = T * cool;

			}

			if (utilitySpace.getUtility(bid) < bias) {
				// System.out.println("**reSA");

				countReSA = loop_count + 1;
				// double t_cool = (0.999 - cool)/20.0;
				if (loop_count > 5) {

					return getBidLowSA(bid, 10000, 0.98, bias, loop_count);
				}
				// System.out.println("**"+ loop_count);
				return getBidHighClimb(bid, loop, bias, loop_count + 1);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// System.out.println(count);
		if (loop_count > 0) {
			// System.out.println("loop_count: "+loop_count);
		}

		return preOffer(bid);
	}

	/**
	 * SA–@‚É‚æ‚éBid‚Ì’T�õƒ�ƒ\ƒbƒh
	 * 
	 * @param bid
	 *            Ží‚Æ‚È‚éBid
	 * @return
	 */
	public Bid getBidSA(Bid bid) {

		// �Ä‚«“Ý‚µ–@
		// Bid bid = utilitySpace.getDomain().getRandomBid();//
		// ƒ‰ƒ“ƒ_ƒ€‚ÉBid‚ð�¶�¬
		// System.out.println(bid.getValues());

		try {

			double T = 10000;// 10000
			int step = 1;// •Ï�X‚·‚é•�
			double cool = 0.999;// ‰·“x‚ð‰º‚°‚é•�
			Random rnd = new Random();

			while (T > 0.0001) {
				double bf = utilitySpace.getUtility(bid);
				Bid bidChange = new Bid(bid);

				// bid change

				int change_id = rnd.nextInt(bid.getIssues().size());// •Ï�X‚·‚é
																	// IssueId‚ÌƒCƒ“ƒfƒbƒNƒX‚ðŒˆ’è
				IssueInteger issueInteger = (IssueInteger) bid.getIssues()
						.get(change_id);
				int issueId = issueInteger.getNumber();
				ValueInteger issueValue = (ValueInteger) bid.getValue(issueId);
				int issueValueInt = Integer.valueOf(issueValue.toString())
						.intValue();
				int max = issueInteger.getUpperBound();// �Å‘å’l
				int min = issueInteger.getLowerBound();// �Å�¬’l

				int flag = rnd.nextBoolean() ? 1 : -1;// 1=> plus , -1 => minus
				if (issueValueInt >= min && issueValueInt <= max) {

					if (issueValueInt + step > max) {
						// ‘«‚·‚Æ–â‘è‚ ‚è
						flag = -1;
					} else if (issueValueInt - step < min) {
						// ˆø‚­‚Æ–â‘è‚ ‚è
						flag = 1;
					}

				}

				Value valueInteger = new ValueInteger(
						issueValueInt + flag * step);// change
														// value
				bidChange = bidChange.putValue(issueId, valueInteger);// change

				double af = utilitySpace.getUtility(bidChange);

				double bf_cost = 1.0 - bf;
				double af_cost = 1.0 - af;

				double p = Math.pow(Math.E, -Math.abs(af_cost - bf_cost) / T);
				// System.out.println(" p = "+p);
				// System.out.println(" rnd = "+rnd.nextDouble());
				if ((af_cost < bf_cost || rnd.nextDouble() < p)
						&& af <= MINIMUM_BID_UTILITY_OFFER) {
					bid = new Bid(bidChange);
				}
				NumberFormat format = NumberFormat.getInstance();
				format.setMaximumFractionDigits(1);

				T = T * cool;

			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// String saveValue = "";
		// for (Entry<Integer, Value> e : bid.getValues().entrySet()) {
		// // System.out.println(" e: "+ e.getValue());
		// saveValue += e.getValue().toString();
		// }

		// System.out.println("e:"+saveValue);

		// if(new Double(format.format(af)) == 1){

		// if (!bidList10.contains(saveValue)) {
		// bidList10.add(saveValue);
		// System.out.println(bidList10.size());
		// } else {
		// System.out.println("ERRRRRORRORORRORRORORORO");
		// }
		// }
		return (bid);
	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}

}

class WhaleSessionData implements Serializable {
	Bid accesptedBid;
	double acceptTime;
	double getUtil;
	double sumUtil;
	int actionType;
	ArrayList<Bid> bidLists;
	Bid offeredBid;
	boolean isAgreement;

	public WhaleSessionData(Bid accesptedBid, Bid offeredBid, double getUtil,
			boolean isAgreement, ArrayList<Bid> bidLists, double sumUtil,
			int actionType) {
		this.accesptedBid = accesptedBid;
		this.offeredBid = offeredBid;
		this.getUtil = getUtil;
		this.isAgreement = isAgreement;
		this.bidLists = bidLists;
		this.sumUtil = sumUtil;
		this.actionType = actionType;
	}
}

//
// class WhaleSessionDataAll implements Serializable{
// ArrayList<WhaleSessionData> All;
// public WhaleSessionDataAll(WhaleSessionData add){
// this.All.add(add);
// }
//
// public void add(WhaleSessionData add) {
// this.All.add(add);
// }
// public WhaleSessionData get(int index){
// return this.All.get(index);
// }
//
// }
/**
 * Map ‚Ì value
 * ‚Åƒ\�[ƒg‚·‚é‚½‚ß‚Ì”äŠr‚ÌƒNƒ‰ƒX
 */
class MapComparator implements Comparator<Bid> {
	private HashMap<Bid, Double> map;

	public MapComparator(HashMap<Bid, Double> map) {
		this.map = map;
	}

	/**
	 * key 2‚Â‚ª—^‚¦‚ç‚ê‚½‚Æ‚«‚É�A‚»‚Ì
	 * value ‚Å”äŠr
	 */
	@Override
	public int compare(Bid key1, Bid key2) {
		// value ‚ðŽæ“¾
		double value1 = map.get(key1);
		double value2 = map.get(key2);
		// value ‚Ì�~�‡, value‚ª“™‚µ‚¢‚Æ‚«‚Í key
		// ‚ÌŽ«�‘�‡
		if (value1 == value2)
			return 1;
		else if (value1 < value2)
			return 1;
		else
			return -1;
	}
}
