package agents.anac.y2014.KGAgent;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import agents.anac.y2014.kGA_gent.library_genetic.GA_Main;
import agents.anac.y2014.kGA_gent.library_genetic.Gene;
import genius.core.Agent;
import genius.core.Bid;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.ActionWithBid;
import genius.core.actions.Offer;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.NonlinearUtilitySpace;

public class KGAgent extends Agent {

	@Override
	public String getVersion() {
		return "5.1.1";
	}

	private Action actionOfPartner = null;
	private static double MINIMUM_UTILITY = 0.0;
	private boolean initflag = true;

	private double acceptableline = 1.0;
	private double acceptderta = 0.001;

	private double time = 0.0;

	private double enemymax = 0.1;

	private double gunma_min = 0.25;
	private double gunma_d = 3;

	List<MyBidGene> bidlist = new ArrayList<MyBidGene>(40);

	History history;

	// TreeSet<MyBidUtil> bidlist = new TreeSet<MyBidUtil>(new CompMyBidUtil());

	int type = 0;

	double timepressur;
	double reservation;

	Bid enemymaxbid = null;

	@Override
	public void init() {
		MINIMUM_UTILITY = utilitySpace.getReservationValueUndiscounted();
		System.out.println("Minimum bid utility: " + MINIMUM_UTILITY);

		reservation = utilitySpace.getReservationValueUndiscounted();
		timepressur = utilitySpace.getDiscountFactor();
		System.out.println(
				"reservate = " + reservation + "   timepress = " + timepressur);

		if (utilitySpace instanceof AdditiveUtilitySpace) {
			type = 0;
			System.out.println("UtilitySpaceType = LINEAR");
		} else if (utilitySpace instanceof NonlinearUtilitySpace) {
			type = 0;
			System.out.println("UtilitySpaceType =  NONLINEAR");
		} else {
			System.out.println("NULL ??  " + utilitySpace.getClass());
			type = 1;
		}
		System.out.println("Type is" + type);

	}

	@Override
	public String getName() {
		return "kGA_gent";
	}

	@Override
	public void ReceiveMessage(Action opponentAction) {
		actionOfPartner = opponentAction;
	}

	int count = 99999;

	@Override
	public Action chooseAction() {
		// TODO
		// è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–
		Action action = null;
		time = timeline.getTime();

		if (initflag) {
			MyInit();
			initflag = false;
		}

		if (actionOfPartner instanceof Offer) {

			Bid enemy = ((Offer) actionOfPartner).getBid();

			if (enemymaxbid == null) {
				enemymaxbid = enemy;
			} else {

				if (getUtility(enemymaxbid) < getUtility(enemy)) {
					enemymaxbid = enemy;
				}

			}

			history.Input(((Offer) actionOfPartner).getBid());

			// System.out.println("EnemyBid is "
			// +getUtility(((Offer)actionOfPartner).getBid()) +
			// " AcceptableLine is " + acceptableline );

			if (Accept(((Offer) actionOfPartner).getBid())) {
				System.out.println("Accept!!!");
				action = new Accept(getAgentID(), enemy);
				return action;
			}

		}

		if (count > 10) {

			// bidlist = SearchBid();

			// System.out.println("GAStart1");
			bidlist = SearchBid2();

			count = 0;

		}

		try {

			// ai.PrintBestFieldData();

			// Bid bid = new
			// Bid(utilitySpace.getDomain(),ai.GetRandomMap(count/100));

			Bid bid = bidlist.get(count).bid;

			if (enemymaxbid != null) {
				if (getUtility(bid) < getUtility(enemymaxbid)) {
					bid = new Bid(enemymaxbid);
				}
			}

			// System.out.println("Bid Utility is" + getUtility(bid));
			acceptableline = Math.min(getUtility(bid), acceptableline);
			action = new Offer(getAgentID(), bid);
		} catch (Exception e) {
			System.out.println("Errer in CooseAction ");
			System.out.println("Exception in ChooseAction:" + e.getMessage());
			// best guess if things go wrong.
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}

		if (action == null) {
			System.out.println("BidListErrer");
			action = new Accept(getAgentID(),
					((ActionWithBid) actionOfPartner).getBid());
		}
		count++;
		return action;
	}

	MyBidGene st;

	void MyInit() {
		// MyHyperRectangle input = new
		// MyHyperRectangle(utilitySpace.getDomain().getIssues());

		// NonlinearUtilitySpace nonlinear = new
		// NonlinearUtilitySpace(utilitySpace);
		/*
		 * ai = new
		 * MyRectangleField(utilitySpace.getDomain().getIssues(),nonlinear
		 * .getAllInclusiveConstraints()); ai.Search();
		 */

		history = new History(utilitySpace.getDomain().getIssues(), this, type);

		// bidlist = SearchBid();

		st = new MyBidGene(this);

		return;
	}

	/*
	 * é�ºä¼�å­�ã�«utilityã‚’ã‚»ãƒƒãƒˆã�
	 * ™ã‚‹
	 */
	public double GetGeneUtility(MyBidGene gene) {
		return getUtility(gene.bid);
	}

	public double GetGeneEnemyUtility(MyBidGene gene) {
		return history.GetEnemyUtility(gene.bid);
	}

	public double EnemyPressAll(double util) {
		return EnemyTimePress(EnemyMaxPress(util));
	}

	ArrayList<MyBidGene> SearchBid2() {
		ArrayList<MyBidGene> ret = new ArrayList<MyBidGene>(40);

		st = new MyBidGene(this);

		enemymax = Math.max(history.SearchMaxPoint(), enemymax);

		// System.out.println("EnemyMax is " + enemymax);

		GA_Main ga;

		if (bidlist.size() > 0) {

			List<Gene> in = new ArrayList<Gene>(300);
			for (int i = 0; i < 60; i++) {
				in.add(new MyBidGene(bidlist.get(i).bid));
			}
			for (int i = 0; i < 300; i++) {
				in.add(new MyBidGene(GetRandomBid()));
			}
			ga = new GA_Main(in, new BidGenerationChange(300, 10),
					new CompMyBidGene(0));

		} else {
			ga = new GA_Main(new BidGenerationChange(300),
					new CompMyBidGene(0));

		}

		// System.out.println("GaStart2");

		ga.Start();

		List<Gene> buf;
		buf = ga.GetList();
		for (int i = 0; i < 60; i++) {

			ret.add((MyBidGene) buf.get(i));
			// System.out.println("Searchdata MyBid " +
			// ((MyBidGene)buf.get(i)).util + " enemybid " +
			// ((MyBidGene)buf.get(i)).enemyutil);
			// System.out.println("in press MyBid " +
			// getUtility(((MyBidGene)buf.get(i)).bid) + " enemybid " +
			// EnemyPressAll(((MyBidGene)buf.get(i)).enemyutil) + " AddUtil " +
			// ((MyBidGene)buf.get(i)).GetValue(0));

		}

		return ret;

	}

	/*
	 * å®Œå…¨ã�«ãƒ©ãƒ³ãƒ€ãƒ ã�ªBidã‚’ç�?Ÿæ
	 * ˆ�ã�™ã‚‹
	 */
	Random randomnr = new Random();

	public Bid GetRandomBid() {

		List<Issue> issues = utilitySpace.getDomain().getIssues();
		HashMap<Integer, Value> values = new HashMap<Integer, Value>();

		IssueInteger lIssueInteger;
		Bid bid = null;
		int optionIndex = 0;

		if (type == 0) {

			for (int i = 0; i < issues.size(); i++) {
				lIssueInteger = (IssueInteger) issues.get(i);
				optionIndex = lIssueInteger.getLowerBound()
						+ randomnr.nextInt(lIssueInteger.getUpperBound()
								- lIssueInteger.getLowerBound());
				values.put(lIssueInteger.getNumber(),
						new ValueInteger(optionIndex));
			}
		} else {
			for (int i = 0; i < issues.size(); i++) {
				IssueDiscrete ID = (IssueDiscrete) issues.get(i);
				int number = randomnr.nextInt(ID.getNumberOfValues());
				values.put(ID.getNumber(), ID.getValue(number));
			}
		}

		try {
			bid = new Bid(utilitySpace.getDomain(), values);
		} catch (Exception e) {
			System.out.println("Enner in GetRandomBid in kGA_gent");
			System.out.println("Exception in ChooseAction:" + e.getMessage());

		}

		return bid;
	}

	double TimePress(double utility) {

		return Math.pow(utility, 2.0 - time);

	}

	double EnemyTimePress(double utility) {

		return utility * 1.3 * Math.pow(time,
				(Math.pow(timepressur * gunma_d, 2.0)) + gunma_min);

		// return utility * (time*1.2 - 0.2);

	}

	double EnemyMaxPress(double util) {

		if (enemymax < 0.001) {
			return 1;
		}

		double ret = util / enemymax;

		if (ret > 1) {
			enemymax = util;
			return 1.0;
		}
		return ret;

	}

	Boolean Accept(Bid partnerbid) {

		double eutil = getUtility(partnerbid);
		if (acceptableline - acceptderta < eutil) {
			System.out.println("Accept!! for acceptableline");
			return true;
		}
		/*
		 * if( eutil > (1.0-time)*0.3 + 0.7 ){
		 * System.out.println("Accept!! for pointhugher"); return true; }
		 */
		return false;
	}

	void BidPrint(Bid bid) {
		List<Issue> issues = bid.getIssues();
		IssueInteger lIssueInteger;

		for (int i = 0; i < issues.size(); i++) {

			lIssueInteger = (IssueInteger) issues.get(i);
			try {
				System.out.print(bid.getValue(lIssueInteger.getNumber()) + " ");
			} catch (Exception e) {
				// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿ
				// catch ãƒ–ãƒ­ãƒƒã‚¯
				e.printStackTrace();
			}
		}
		System.out.println();

	}

	@Override
	public double getUtility(Bid bid) {

		try {
			return utilitySpace.getUtility(bid);
		} catch (Exception e) {
			// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿ catch
			// ãƒ–ãƒ­ãƒƒã‚¯
			e.printStackTrace();

		}
		return 0;

	}

	@Override
	public String getDescription() {
		return "ANAC2014 compatible with non-linear utility spaces";
	}

}
