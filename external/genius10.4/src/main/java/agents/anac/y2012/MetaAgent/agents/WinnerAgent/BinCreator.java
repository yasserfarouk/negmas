package agents.anac.y2012.MetaAgent.agents.WinnerAgent;


import java.util.ArrayList;
import java.util.Vector;

import genius.core.issue.Value;

public abstract class BinCreator {
	int numOfBins;
	double percentageOfRange = 0.02;
	int numConst = 5;
	public abstract ArrayList<DiscretisizedKey> createBins(double min, double max);
	public abstract Vector<? extends Value> createValuesVector(double min, double max);
}