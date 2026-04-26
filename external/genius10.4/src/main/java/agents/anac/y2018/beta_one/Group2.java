package agents.anac.y2018.beta_one;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.math3.stat.regression.SimpleRegression;

import genius.core.AgentID;
import genius.core.actions.Offer;
import genius.core.list.Tuple;
import genius.core.misc.Range;
import genius.core.persistent.StandardInfo;
import genius.core.persistent.StandardInfoList;
import genius.core.utility.AdditiveUtilitySpace;

public class Group2 extends GroupNegotiator
{
	private static final double OBSERVE_DURATION = 0.75;
	private static final double BOX_SIZE = 0.05;
	private static final double SLOPE_TOLERANCE = 0.035;

	private static final double MIN_SELFISH_RATIO = 0.15;
	private static final double MAX_SELFISH_RATIO = 0.30;

	private static final int HISTORY_ANALYZE_COUNT = 5;

	private AntiAnalysis antiAnalysis;

	private HashMap<AgentID, SimpleRegression> data;

	@Override
	protected void initialize()
	{
		antiAnalysis = new AntiAnalysis((AdditiveUtilitySpace) utilitySpace, OBSERVE_DURATION, (MIN_SELFISH_RATIO + MAX_SELFISH_RATIO) / 2.0, BOX_SIZE);

		data = new HashMap<AgentID, SimpleRegression>();
	}

	@Override
	protected void initializeHistory(StandardInfoList infoList)
	{
		if (infoList.isEmpty())
			return;

		Map<String, Double> utilitySet = new HashMap<String, Double>();
		for (int i = infoList.size() - 1; i >= 0 && i >= infoList.size() - HISTORY_ANALYZE_COUNT; i--)
		{
			StandardInfo info = infoList.get(i);

			for (Tuple<String, Double> offered : info.getUtilities())
			{
				String agent = offered.get1();
				Double utility = offered.get2();

				agent = agent.substring(0, agent.indexOf("@"));

				if (!utilitySet.containsKey(agent) || utility < utilitySet.get(agent).doubleValue())
					utilitySet.put(agent, utility);
			}
		}

		double maxUtility = 0;

		for (Entry<String, Double> entry : utilitySet.entrySet())
		{
			if (maxUtility < entry.getValue())
				maxUtility = entry.getValue();
		}

		double selfishRatio = AntiAnalysis.lerp(MIN_SELFISH_RATIO, MAX_SELFISH_RATIO, maxUtility);
		antiAnalysis.setSelfishPoint(selfishRatio);
	}

	@Override
	public void receiveOffer(Offer receivedOffer, double utility)
	{
		AgentID id = receivedOffer.getAgent();
		addData(id, utility);
		boolean betrayed = hasBetrayed(id);

		double t = negotiationTime / utilitySpace.getDiscountFactor();
		Range range = antiAnalysis.getBox(t, betrayed);
		setAcceptableRange(id, range);
	}

	private boolean hasBetrayed(AgentID id)
	{
		double mySlope = antiAnalysis.getSelfishSlope(utilitySpace.getDiscountFactor());
		double oppSlope = getSlope(id);

		return oppSlope + mySlope < -SLOPE_TOLERANCE;
	}

	private double getSlope(AgentID id)
	{
		return getRegression(id).getSlope();
	}

	private void addData(AgentID id, double utility)
	{
		getRegression(id).addData(negotiationTime, utility);
	}

	private SimpleRegression getRegression(AgentID id)
	{
		if (!data.containsKey(id))
		{
			data.put(id, new SimpleRegression());
		}

		return data.get(id);
	}
}
