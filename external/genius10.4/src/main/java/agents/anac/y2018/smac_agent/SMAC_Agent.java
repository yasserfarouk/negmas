package agents.anac.y2018.smac_agent;

import java.util.List;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.Inform;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.SortedOutcomeSpace;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Value;
import genius.core.misc.Range;
import genius.core.parties.AbstractNegotiationParty;
import genius.core.parties.NegotiationInfo;
import genius.core.utility.AbstractUtilitySpace;

public class SMAC_Agent extends AbstractNegotiationParty {

	// Parameter initialisation
	private Bid lastReceivedBid = null;
	private Bid myLastBid = null;
	private SortedOutcomeSpace sortedOutcomeSpace = null;
	private UtilCurve BS_curve = null;
	private UtilCurve AS_curve = null;
	private UtilCurve OMS_curve = null;

	private double resValue = 0;
	private double discountFactor = 0;
	private double domainSize = 0;
	private double potLoss = 0;

	private int domainSmallThres = 100;
	private int domainMediumThres = 1000;
	private double resThres = 0.5;
	private double discountThres = 0.5;
	private int partiesToReveal = -1;
	private int round = 0;

	private Map<String, AgentProfile> agentProfiles = new HashMap<>();

	private double BS_utilInit;
	private double BS_utilEnd;
	private double BS_shapeLeft;
	private double BS_shapeRight;
	private double BS_stdDev;
	private double BS_utilRange;
	private int BS_useUtilRange;
	private int BS_scaleMinMax;

	private double AS_timethreshold1;
	private double AS_timethreshold2;
	private double AS_timethreshold3;
	private double AS_rate1;
	private double AS_rate2;
	private double AS_rate3;
	private double AS_discount_rate1;
	private double AS_discount_rate2;
	private double AS_discount_rate3;
	private double AS_threshold1;

	private double OMS_utilInit;
	private double OMS_utilEnd;
	private double OMS_shapeLeft;
	private double OMS_shapeRight;
	private int OMS_useScaling;
	private int OMS_useOM;
	private int OMS_limitScaling;
	private int OMS_euclidean;

	private int useResValue;
	private int RM_updateUtilCurve;

	// Arrays of optimised parameters to select config from
	private double[] BS_utilInit_Array = { 0.8726814917149487, 0.777828754759162, 0.794611415535526, 0.8338477204370194,
			0.794611415535526, 0.9902714777558712, 0.8177264028188669, 0.9369503033177021, 0.8752031270267119,
			0.9130860700146172, 0.8985462879619349, 0.968496462717577, 0.9475036977013452, 0.6358476408988469,
			0.8187434422624748 };
	private double[] BS_utilEnd_Array = { 0.7234378711390601, 0.5039748507682985, 0.28288734275992156,
			0.5043866091578793, 0.28288734275992156, 0.4003485368855013, 0.5775989724689408, 0.8449151525540153,
			0.6439901974110889, 0.5128376365334515, 0.8815381590699309, 0.7512965956758736, 0.7732741878711409,
			0.6890153697768392, 0.4575950773478153 };
	private double[] BS_shapeLeft_Array = { -3.3591096890191987, -1.701604675744841, -1.477010357010462,
			-3.8655124599356254, -1.477010357010462, -4.578937298449693, -0.6220720689296879, -0.33672176040647983,
			-1.028771048186937, -2.0559509070865096, -2.8918016435189267, -0.7799716189241472, -1.3695282977990084,
			-5.218031753259722, -2.751303202136022 };
	private double[] BS_shapeRight_Array = { 1.6257316909626451, 5.743207917861144, 0.5042105201178133,
			1.2177749917981506, 0.5042105201178133, 0.23509522311374287, 4.091234186082431, 4.1350076547071914,
			2.4754885878922317, 2.164288746994306, 2.7254163735351717, 4.6865277838004795, 3.2756425247103667,
			4.954211560087625, 0.9268696757205295 };
	private double[] BS_stdDev_Array = { 0.011291090432680746, 0.00806602340055852, 0.03993171716605801,
			0.04541686791307292, 0.03993171716605801, 0.016756394004539998, 0.02050154318077101, 0.029271019497824725,
			0.043415206320189886, 0.0070856192273503205, 0.039755330055406116, 0.016399241309376957,
			0.025102281569543264, 0.04820460835411705, 0.02050857459278594 };
	private double[] BS_utilRange_Array = { 0.12747022094357666, 0, 0, 0, 0, 0.13272162785215005, 0, 0.0983190266452012,
			0.05856974543348425, 0, 0.07746667826547407, 0.07280579445258521, 0.18730438418792866, 0.14765559656825974,
			0.1 };
	private int[] BS_useUtilRange_Array = { 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1 };
	private int[] BS_scaleMinMax_Array = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	private double[] AS_timethreshold1_Array = { 0.04674955360732539, 0.21892573208596458, 0.16179316499336657,
			0.017950789417510913, 0.16179316499336657, 0.24113343505148666, 0.07195359710638492, 0.24062799038166227,
			0.06459440421448916, 0.05992314378385267, 0.0404833611227041, 0.016365548306668937, 0.03720304384144093,
			0.03478252915311665, 0.09717022719016756 };
	private double[] AS_timethreshold2_Array = { 0.43303206747881884, 0.43815743675678354, 0.661647912300392,
			0.6641557212474182, 0.661647912300392, 0.6343062109766446, 0.31691596321099397, 0.6193435123380937,
			0.5101124045982374, 0.6439117927197825, 0.3097759359131537, 0.6779137891044901, 0.3191532652234305,
			0.36151362475862164, 0.49570228758309676 };
	private double[] AS_timethreshold3_Array = { 0.7245716275829817, 0.9673408842892797, 0.9430960906851461,
			0.8068138909742011, 0.9430960906851461, 0.9259546534020413, 0.7785900567996049, 0.7315861505357254,
			0.9861492577603947, 0.8404594822840576, 0.8996360620044062, 0.8866203037044917, 0.7572372266466516,
			0.9713328720627714, 0.7682567113937757 };
	private double[] AS_rate1_Array = { 0.11219220522147239, 0.08003925149326611, 0.33051281601825294,
			0.5168640464469102, 0.33051281601825294, 0.13512460983555102, 0.3688780153278271, 0.29676619243083197,
			0.6646699560793469, 0.08650858607226897, 0.5000849837001996, 0.6245922813980603, 0.40987041296162613,
			0.5267683830657954, 0.42081900454556703 };
	private double[] AS_rate2_Array = { 0.004898877513807327, 0.5634389184435432, 0.5882550637609057,
			0.1736660799531687, 0.5882550637609057, 0.3052688029644835, 0.2634165350102941, 0.1128289859613561,
			0.09148496643943949, 0.594472202286485, 0.003764210630213882, 0.3267176154876564, 0.16025942852106756,
			0.33889339364499194, 0.29231207452138 };
	private double[] AS_rate3_Array = { 0.548875357115968, 0.4749109394742966, 0.14967914649484768, 0.12365288690420985,
			0.14967914649484768, 0.46068703380168724, 0.008086833565920028, 0.4505848794497385, 0.5688130910774031,
			0.009397472597385702, 0.3072190282216138, 0.013929751760161762, 0.22948249946166988, 0.24981417826220348,
			0.02859766493418777 };
	private double[] AS_discount_rate1_Array = { 0.60078421520277, 0.041443536726597396, 0.062408258801741855,
			0.09097496706566402, 0.062408258801741855, 0.2201074114341298, 0.6400322557728236, 0.5788620610232695,
			0.11081606927344186, 0.5229841600644036, 0.24433177863341993, 0.4661668394745566, 0.35628521577122196,
			0.33720834099081726, 0.049371938906709045 };
	private double[] AS_discount_rate2_Array = { 0.60078421520277, 0.041443536726597396, 0.062408258801741855,
			0.09097496706566402, 0.062408258801741855, 0.2201074114341298, 0.6400322557728236, 0.5788620610232695,
			0.11081606927344186, 0.5229841600644036, 0.3208622393478284, 0.4661668394745566, 0.35628521577122196,
			0.33720834099081726, 0.049371938906709045 };
	private double[] AS_discount_rate3_Array = { 0.7657680975742767, 0.39061103724559, 0.2228029756845122,
			0.23097016208565402, 0.2228029756845122, 0.6512371835270208, 0.6855684205205383, 0.051066815285715085,
			0.3292575463583758, 0.5720037378648264, 0.40017902857747045, 0.2710384577710453, 0.25373107722701543,
			0.36883611624486773, 0.10017027025202459 };
	private double[] AS_threshold1_Array = { 0.8219843158180532, 0.923099025516942, 0.7932091636158838,
			0.9580958735836536, 0.7932091636158838, 0.9685239192034572, 0.7859975846053607, 0.7499410639013572,
			0.7020301444423871, 0.7739229197734587, 0.9240012133461064, 0.7794051769942631, 0.8620257399086155,
			0.7607469411657954, 0.8870099642076714 };

	private double[] OMS_utilInit_Array = { 0, 0, 0, 0, 0, 0, 0, 0.815223752226039, 0, 0, 0, 0, 0, 0,
			0.5163591000775694 };
	private double[] OMS_utilEnd_Array = { 0, 0, 0, 0, 0, 0, 0, 0.6473775354471122, 0, 0, 0, 0, 0, 0,
			0.7778975091828539 };
	private double[] OMS_shapeLeft_Array = { 0, 0, 0, 0, 0, 0, 0, -1.7202463296555042, 0, 0, 0, 0, 0, 0,
			-0.5137389957405025 };
	private double[] OMS_shapeRight_Array = { 0, 0, 0, 0, 0, 0, 0, 2.317670771610482, 0, 0, 0, 0, 0, 0,
			5.911300168932384 };
	private int[] OMS_useScaling_Array = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
	private int[] OMS_useOM_Array = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1 };
	private int[] OMS_limitScaling_Array = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	private int[] OMS_euclidean_Array = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	private int[] useResValue_Array = { 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1 };
	private int[] RM_updateUtilCurve_Array = { 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0 };
	// ----------

	@Override
	public void init(NegotiationInfo info) {
		super.init(info);

		sortedOutcomeSpace = new SortedOutcomeSpace(utilitySpace);

		// Sets max and min utility in domain
		double utilMin = sortedOutcomeSpace.getMinBidPossible().getMyUndiscountedUtil();
		double utilMax = sortedOutcomeSpace.getMaxBidPossible().getMyUndiscountedUtil();

		// Obtain res' value, discounted value and domain size
		resValue = utilitySpace.getReservationValue();
		discountFactor = utilitySpace.getDiscountFactor();
		domainSize = utilitySpace.getDomain().getNumberOfPossibleBids();

		// Set agent parameters accordingly
		initparameter(resValue, discountFactor, domainSize);

		// Set minimum utility to reservation value if higher
		if (resValue > utilMin && useResValue == 1) {
			utilMin = resValue;
		}

		// Initialise ulitity curves
		BS_curve = new UtilCurve(BS_shapeLeft, BS_shapeRight, BS_utilInit, BS_utilEnd);
		OMS_curve = new UtilCurve(OMS_shapeLeft, OMS_shapeRight, OMS_utilInit, OMS_utilEnd);

		// Change bidding utility parameters
		if (BS_scaleMinMax == 1) {
			BS_curve.setMinMax(utilMin, utilMax);
		}
		BS_curve.setStdDev(BS_stdDev);

	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> validActions) {
		round++;

		if (round > 1 && AS() == 1) {
			return new Accept(getPartyId(), lastReceivedBid);
		} else {
			myLastBid = determineNextBid();
			return new Offer(getPartyId(), myLastBid);
		}
	}

	public int AS() {
		// get some values
		double time = timeline.getTime();
		double nextMyBidUtil = getUtilityWithDiscount(myLastBid);
		double lastOpponentBidUtil = getUtilityWithDiscount(lastReceivedBid);

		// Check first threshold
		if (lastOpponentBidUtil >= AS_threshold1) {
			return 1;
		}

		if (lastOpponentBidUtil < resValue * discountFactor) {
			return 0;
		}
		// time=(0, time_threshold1]
		if (time <= AS_timethreshold1) {
			return 0;
		} else if (time <= AS_timethreshold2) { // time=(time_threshold1, time_threshold2]
			if (lastOpponentBidUtil > AS_threshold1 - (time - AS_timethreshold1) * AS_rate1
					- (1 - discountFactor) * (time - AS_timethreshold1) * AS_discount_rate1) {
				return 1;
			} else
				return 0;
		} else if (time <= AS_timethreshold3) { // time=(time_threshold2, time_threshold3]
			if (lastOpponentBidUtil > AS_threshold1 - (AS_timethreshold2 - AS_timethreshold1) * AS_rate1 / 2
					- Math.pow(10 * (time - AS_timethreshold2), 2) / 10 * AS_rate2
					- (1 - discountFactor) * (time - AS_timethreshold2) * AS_discount_rate2) {
				return 1;
			} else
				return 0;
		}
		// time=(time_threshold3,1]
		else {
			if (lastOpponentBidUtil > nextMyBidUtil - (time - AS_timethreshold3) * AS_rate3
					- (1 - discountFactor) * (time - AS_timethreshold3) * AS_discount_rate3) {
				return 1;
			}
		}

		return 0;
	}

	public Bid determineNextBid() {
		// Initialise values
		double time = timeline.getTime();
		Range range = null;
		double util = BS_curve.getUtil(time);

		// Max util = 1
		if (util > 1)
			util = 1;

		// Sets utility range to go for
		if (BS_useUtilRange == 1) {
			range = new Range(util, util + BS_utilRange);
		} else {
			range = new Range(util, 1);
		}

		// Selects bid based on utility thresholds, expands threshold if no bids are
		// found
		BidDetails bidDetails = null;
		while (bidDetails == null) {
			List<BidDetails> lBids = sortedOutcomeSpace.getBidsinRange(range);
			bidDetails = OMS(lBids);
			if (bidDetails == null) {
				range.setLowerbound(range.getLowerbound() - 0.001);
				range.setUpperbound(range.getUpperbound() + 0.001);
			}
		}
		return bidDetails.getBid();
	}

	public BidDetails OMS(List<BidDetails> allBids) {
		// Initialise values
		double time = timeline.getTime();
		double utilGoal = OMS_curve.getUtil(time);

		// 1. If there is only a single bid, return this bid
		if (allBids.size() == 1) {
			System.out.println("OMS_ found only 1 bid");
			return null;
		}

		// If no OM is used, return random bid
		if (OMS_useOM == 0) {
			return allBids.get(rand.nextInt(allBids.size()));
		}

		// Initialise individual opponent util
		Map<String, Double> agentUtilGoals = new HashMap<>();

		
		if (OMS_useScaling == 1 && partiesToReveal == 0) {
			// Set individual opponent util according to their thoughness
			double sum = 0.0;
			double count = 0.0;
			for (String key : agentProfiles.keySet()) {
				agentUtilGoals.put(key, agentProfiles.get(key).getLastBidOppUtil());
				sum += agentUtilGoals.get(key);
				count++;
			}
			double average = sum / count;
			for (String key : agentUtilGoals.keySet()) {
				double util = agentUtilGoals.get(key) - average + utilGoal;
				if (util > 1 && OMS_limitScaling == 1) {
					agentUtilGoals.put(key, 1.0);
				} else {
					agentUtilGoals.put(key, util);
				}
			}
		} else {
			// Set individual opponent util to the same value
			for (String key : agentProfiles.keySet()) {
				agentUtilGoals.put(key, utilGoal);
			}
		}

		double diff = 2;
		BidDetails nearestBid = allBids.get(allBids.size() - 1);

		// Select bid from list based on distance to individual opponent util
		for (BidDetails bidDetails : allBids) {
			double minDistance = 0.0;
			double distance = 0.0;
			boolean validBid = true;
			for (String key : agentUtilGoals.keySet()) {
				double oppUtilGoal = agentUtilGoals.get(key);
				double oppUtil = agentProfiles.get(key).evaluateBid(bidDetails.getBid());
				if (oppUtil > oppUtilGoal) {
					validBid = false;
					break;
				} else {
					if (OMS_euclidean == 1) {
						//euclidean
						distance += Math.pow(oppUtilGoal - oppUtil, 2);
					} else {
						//chebyshev
						distance = Math.abs(oppUtilGoal - oppUtil);
						if (distance > minDistance) {
							minDistance = distance;
						}
					}
				}
			}
			if (OMS_euclidean == 1) {
				minDistance = Math.sqrt(distance);
			}

			// Save minimum distance
			if (minDistance < diff && validBid) {
				nearestBid = bidDetails;
				diff = minDistance;
			}
		}
		return nearestBid;
	}

	@Override
	public void receiveMessage(AgentID sender, Action action) {

		if (action instanceof Offer) {
			lastReceivedBid = ((Offer) action).getBid();
			String agentName = getPartyName(action.getAgent().toString());

			// initialise opponent model
			if (agentProfiles.get(agentName) == null) {
				partiesToReveal--;
				agentProfiles.put(agentName, new AgentProfile(utilitySpace));
			}
			agentProfiles.get(agentName).registerOffer(lastReceivedBid);
		} else if (action instanceof Accept) {
			if (RM_updateUtilCurve == 1 && partiesToReveal == 0) {
				// Update utility curve based on accepting util
				Bid bid = ((Accept) action).getBid();
				String agentName = getPartyName(action.getAgent().toString());

				agentProfiles.get(agentName).registerAccept(bid);
				updateUtilCurve();
			}
		} else if (action instanceof Inform) {
			partiesToReveal = (int) ((Inform) action).getValue() - 1;
		}
	}

	private void updateUtilCurve() {
		double minAcceptedUtil = 1.0;
		for (String key : agentProfiles.keySet()) {
			double acceptedUtil = getUtility(agentProfiles.get(key).getBestAccepted());
			if (acceptedUtil < minAcceptedUtil) {
				minAcceptedUtil = acceptedUtil;
			}
		}
		BS_curve.setMinAccepted(minAcceptedUtil);
	}

	private String getPartyName(String partyID) {
		return partyID.substring(0, partyID.indexOf("@"));
	}

    @Override
    public String getDescription() {
        return "ANAC2018";
    }

	// This function selects the parameter set based on the domain
	void initparameter(double resValue, double discountFactor, double domainSize) {
		int resdisIndex = 0;
		int domainIndex = 0;
		int parameterIndex = 0;

		// Parameter selection
		// ##################
		// ##################
		if (resValue == 0 || discountFactor == 1) {
			resdisIndex = 0;
		} else if (resValue <= resThres && discountFactor <= discountThres) {
			resdisIndex = 1;
		} else if (resValue <= resThres && discountFactor > discountThres) {
			resdisIndex = 2;
		} else if (resValue > resThres && discountFactor <= discountThres) {
			resdisIndex = 3;
		} else if (resValue > resThres && discountFactor > discountThres) {
			resdisIndex = 4;
		}

		if (domainSize <= domainSmallThres) {
			domainIndex = 0;
		} else if (domainSize <= domainMediumThres) {
			domainIndex = 1;
		} else {
			domainIndex = 2;
		}

		parameterIndex = 5 * domainIndex + resdisIndex;
		System.out.println("parameterIndex:" + parameterIndex);

		// Assign parameter value
		// #############################
		// #############################

		BS_utilInit = BS_utilInit_Array[parameterIndex];
		BS_utilEnd = BS_utilEnd_Array[parameterIndex];
		BS_shapeLeft = BS_shapeLeft_Array[parameterIndex];
		BS_shapeRight = BS_shapeRight_Array[parameterIndex];
		BS_stdDev = BS_stdDev_Array[parameterIndex];
		BS_utilRange = BS_utilRange_Array[parameterIndex];
		BS_useUtilRange = BS_useUtilRange_Array[parameterIndex];
		BS_scaleMinMax = BS_scaleMinMax_Array[parameterIndex];

		AS_timethreshold1 = AS_timethreshold1_Array[parameterIndex];
		AS_timethreshold2 = AS_timethreshold2_Array[parameterIndex];
		AS_timethreshold3 = AS_timethreshold3_Array[parameterIndex];
		AS_rate1 = AS_rate1_Array[parameterIndex];
		AS_rate2 = AS_rate2_Array[parameterIndex];
		AS_rate3 = AS_rate3_Array[parameterIndex];
		AS_discount_rate1 = AS_discount_rate1_Array[parameterIndex];
		AS_discount_rate2 = AS_discount_rate2_Array[parameterIndex];
		AS_discount_rate3 = AS_discount_rate3_Array[parameterIndex];
		AS_threshold1 = AS_threshold1_Array[parameterIndex];

		OMS_utilInit = OMS_utilInit_Array[parameterIndex];
		OMS_utilEnd = OMS_utilEnd_Array[parameterIndex];
		OMS_shapeLeft = OMS_shapeLeft_Array[parameterIndex];
		OMS_shapeRight = OMS_shapeRight_Array[parameterIndex];
		OMS_useScaling = OMS_useScaling_Array[parameterIndex];
		OMS_useOM = OMS_useOM_Array[parameterIndex];
		OMS_limitScaling = OMS_limitScaling_Array[parameterIndex];
		OMS_euclidean = OMS_euclidean_Array[parameterIndex];

		useResValue = useResValue_Array[parameterIndex];
		RM_updateUtilCurve = RM_updateUtilCurve_Array[parameterIndex];

	}

	// Utility curve class
	private class UtilCurve {
		double shapeLeft;
		double shapeRight;
		double utilInit;
		double utilEnd;

		double correction;
		double correctionFactor;

		double utilMin = 0.0;
		double utilMax = 1.0;
		double stdDev = 0.0;

		public UtilCurve(double shapeLeft, double shapeRight, double utilInit, double utilEnd) {
			this.shapeLeft = shapeLeft;
			this.shapeRight = shapeRight;
			this.utilInit = utilInit;
			this.utilEnd = utilEnd;

			correction = 1.0 / (1.0 + Math.exp(-shapeLeft));
			correctionFactor = (utilEnd - utilInit) / (1.0 / (1.0 + Math.exp(-shapeRight)) - correction);
		}

		public void setMinMax(double utilMin, double utilMax) {
			this.utilMin = utilMin;
			this.utilMax = utilMax;
		}

		public void setStdDev(double stdDev) {
			this.stdDev = stdDev;
		}

		public void setMinAccepted(double minAccepted) {
			double finalUtil = utilMin + (utilMax - utilMin) * utilEnd;
			if (minAccepted > finalUtil) {
				utilEnd = (minAccepted - utilMin) / (utilMax - utilMin);
			}
			correctionFactor = (utilEnd - utilInit) / (1.0 / (1.0 + Math.exp(-shapeRight)) - correction);
		}

		public double getUtil(double time) {
			double x = (shapeRight - shapeLeft) * time + shapeLeft;
			double S = (1.0 / (1.0 + Math.exp(-x)) - correction) * correctionFactor + utilInit;

			return utilMin + (utilMax - utilMin) * S + rand.nextGaussian() * stdDev;
		}
	}

	// Opponent model class
	private class AgentProfile {
		private LinkedHashMap<Integer, HashMap<Value, Incremental>> valueFrequency = new LinkedHashMap<>();
		private LinkedHashMap<Integer, Incremental> issueFrequency = new LinkedHashMap<>();

		private Bid lastBid = null;
		private Bid bestAccepted = null;

		private double bidsReceived = 0;
		private double issuesChanged = 0;
		private double issues = 0;

		public AgentProfile(AbstractUtilitySpace utilitySpace) {
			for (Issue issue : utilitySpace.getDomain().getIssues()) {
				issues++;
				Integer issuenr = issue.getNumber();
				issueFrequency.put(issuenr, new Incremental());

				valueFrequency.put(issuenr, new HashMap<>());
				IssueDiscrete issued = (IssueDiscrete) issue;

				for (Value value : issued.getValues()) {
					valueFrequency.get(issuenr).put(value, new Incremental());
				}
			}
		}

		public void registerOffer(Bid bid) {
			bidsReceived++;

			if (lastBid == null)
				lastBid = bid;

			for (Issue issue : bid.getIssues()) {
				Integer issuenr = issue.getNumber();
				Value value = bid.getValue(issuenr);
				valueFrequency.get(issuenr).get(value).increment();

				if (value != lastBid.getValue(issuenr)) {
					issuesChanged++;
					issueFrequency.get(issuenr).increment();
				}
			}
		}

		public void registerAccept(Bid bid) {
			double utility = getUtility(bid);

			if (bestAccepted == null) {
				bestAccepted = bid;
			} else if (utility > getUtility(bestAccepted)) {
				bestAccepted = bid;
			}
		}

		public double evaluateBid(Bid bid) {
			double predict = 0.0;

			if (bidsReceived == 0)
				return 0.0;

			for (Issue issue : bid.getIssues()) {
				Integer issuenr = issue.getNumber();
				Value value = bid.getValue(issuenr);

				double fValue = valueFrequency.get(issuenr).get(value).get();
				double fIssue = issueFrequency.get(issuenr).get();

				if (issuesChanged == 0) {
					predict += (fValue / bidsReceived) * (1.0 / issues);
				} else {
					predict += (fValue / bidsReceived) * (1.0 - fIssue / issuesChanged) / (issues - 1.0);
				}
			}
			return predict;
		}

		public Bid getBestAccepted() {
			return bestAccepted;
		}

		public double getLastBidOppUtil() {
			return evaluateBid(lastBid);
		}

		public String toString() {
			return valueFrequency.toString() + "\n" + issueFrequency.toString();
		}
	}

	private class Incremental {
		int value = 0;

		public void increment() {
			value++;
		}

		public int get() {
			return value;
		}

		public String toString() {
			return String.valueOf(value);
		}
	}
}
