package genius.core.utility;

import java.util.ArrayList;

import genius.core.Bid;

public abstract class HyperRectangle extends Constraint {

	public abstract ArrayList<Bound> getBoundList();

	protected abstract void setBoundList(ArrayList<Bound> boundlist);

	public abstract double getUtilityValue();

	protected abstract void setUtilityValue(double utilityValue);

	public abstract double getUtility(Bid bid);

}
