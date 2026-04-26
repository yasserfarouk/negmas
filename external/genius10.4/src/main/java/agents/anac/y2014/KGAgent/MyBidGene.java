package agents.anac.y2014.KGAgent;

import java.util.List;
import java.util.Random;

import agents.anac.y2014.kGA_gent.library_genetic.Gene;
import genius.core.Bid;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.ValueInteger;

public class MyBidGene implements Gene {

	/**
	 * @param args
	 */
	double mut = 0.1;// çª�ç„¶å¤‰ç•°ç¢ºç«‹

	static KGAgent agent = null;

	static int type = 0;

	double util = 0.0;
	double enemyutil = 0.0;

	boolean utilflag = false;

	Bid bid;

	static Random randomnr = null;

	static boolean agentflag = false;

	MyBidGene(Bid b) {

		bid = new Bid(b);
	}

	public MyBidGene(MyBidGene mybid) {

		bid = new Bid(mybid.bid);
		// util = mybid.util;
		// enemyutil = mybid.enemyutil;

		utilflag = false;

		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿã‚³ãƒ³ã‚¹ãƒˆãƒ©ã‚¯ã‚¿ãƒ¼ãƒ»ã‚¹ã‚¿ãƒ–
	}

	public MyBidGene(KGAgent agent) {
		this.agent = agent;
		randomnr = new Random();
		agentflag = true;
		type = agent.type;
	}

	public MyBidGene() {
		if (agentflag == false) {
			System.out.println("AgentSetErrer in MyBidGene");
		}
		// System.out.println("Call MyBidGene Instance");
		bid = agent.GetRandomBid();
		// System.out.println("End MyBidGene Instance");
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;

		}
		if (obj == null) {
			return false;
		}
		return bid.equals((Bid) obj);
	}

	@Override
	public Gene Cros(Gene g1) {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–

		List<Issue> issues = bid.getIssues();

		Bid in1 = ((MyBidGene) g1).bid;

		Bid ret = new Bid(bid);

		for (int i = 0; i < issues.size(); i++) {

			if (randomnr.nextDouble() < 0.5) {

				try {
					if (type == 0) {
						IssueInteger lIssueInteger = (IssueInteger) issues
								.get(i);
						ret = ret.putValue(lIssueInteger.getNumber(),
								in1.getValue(lIssueInteger.getNumber()));
					} else {
						IssueDiscrete ID = (IssueDiscrete) issues.get(i);
						ret = ret.putValue(ID.getNumber(),
								in1.getValue(ID.getNumber()));

					}

				} catch (Exception e) {
					// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿ catch ãƒ–ãƒ­ãƒƒã‚¯
					e.printStackTrace();
					System.out
							.println("GeneCrossErrer Number Out? in MyBidGene Line80");
				}

			}
		}
		return new MyBidGene(ret);
	}

	public void Mutate() {

		List<Issue> issues = bid.getIssues();

		/*
		 * çª�ç„¶å¤‰ç•°
		 */
		for (int i = 0; i < issues.size(); i++) {

			if (randomnr.nextDouble() < mut) {

				if (type == 0) {
					IssueInteger lIssueInteger = (IssueInteger) issues.get(i);
					int rndint = randomnr.nextInt(lIssueInteger.getUpperBound()
							- lIssueInteger.getLowerBound());
					bid = bid.putValue(lIssueInteger.getNumber(),
							new ValueInteger(lIssueInteger.getLowerBound()
									+ rndint));
				} else {
					IssueDiscrete ID = (IssueDiscrete) issues.get(i);
					int number = randomnr.nextInt(ID.getNumberOfValues());
					bid = bid.putValue(ID.getNumber(), ID.getValue(number));
				}
			}
		}
		utilflag = false;

	}

	/*
	 * state -1 enemy 0 all 1 my
	 */
	public double GetValue(int state) {

		if (agent == null) {
			System.out.println("AgentSetErrer in GetValue in MyBidGene");
		}

		if (utilflag == false) {

			util = agent.GetGeneUtility(this);
			enemyutil = agent.GetGeneEnemyUtility(this);
			utilflag = true;

		}
		switch (state) {
		case -1:
			return enemyutil;
		case 0:
			return agent.EnemyPressAll(enemyutil) + util;
		case 1:
			return util;
		default:
			return 0;
		}
	}

	@Override
	public double GetValue() {
		// TODO è‡ªå‹•ç�?Ÿæˆ�ã�•ã‚Œã�Ÿãƒ¡ã‚½ãƒƒãƒ‰ãƒ»ã‚¹ã‚¿ãƒ–

		if (utilflag == false) {

			util = agent.GetGeneUtility(this);
			enemyutil = agent.GetGeneEnemyUtility(this);
			utilflag = true;

		}
		return agent.getUtility(bid);
	}

	public void SetAgent(KGAgent a) {
		agent = a;
	}

}
