package genius.gui.tree;

import genius.core.listener.Listener;
import genius.core.utility.UncertainAdditiveUtilitySpace;
import genius.core.utility.UtilitySpace;
import genius.gui.panels.BooleanModel;
import genius.gui.panels.DoubleModel;
import genius.gui.panels.IntegerModel;

/**
 * Holds the uncertainty settings (while editing a profile). This MVC model
 * contains listenable fields and coordinates changes
 */
public class UncertaintySettingsModel {
	private final BooleanModel isEnabled;
	private final IntegerModel comparisons;
	private final IntegerModel errors;
	private final DoubleModel  elicitationCost;
	private final BooleanModel isFixedSeed;
	private final BooleanModel isExperimental;
	private final long totalBids;

	/**
	 * @oaran space the {@link UtilitySpace} we're working with. This panel is
	 *        enabled by default iff the space is an
	 *        {@link UncertainAdditiveUtilitySpace}. If the space is
	 *        {@link UncertainAdditiveUtilitySpace} we also copy the default
	 *        values from there. Otherwise all default values are set to 0.
	 */
	public UncertaintySettingsModel(UtilitySpace space) {
		final UncertainAdditiveUtilitySpace uspace = space instanceof UncertainAdditiveUtilitySpace
				? (UncertainAdditiveUtilitySpace) space
				: null;

		isEnabled = new BooleanModel(uspace != null);

		totalBids = space.getDomain().getNumberOfPossibleBids();

		isFixedSeed = new BooleanModel(
				(uspace != null) ? uspace.isFixedSeed() : true);
		
		isExperimental = new BooleanModel(
				uspace != null ? uspace.isExperimental() : false);

		comparisons = new IntegerModel(
				uspace != null ? uspace.getComparisons() : 2, 2, (int) totalBids, 1);

		errors = new IntegerModel(uspace != null ? uspace.getErrors() : 0, 0,
				(int) totalBids, 1);
		
		elicitationCost = new DoubleModel(uspace != null ? uspace.getElicitationCost() : 0);

		// copy enabledness -> lockedness of other components
		isEnabled.addListener(new Listener<Boolean>() {
			@Override
			public void notifyChange(Boolean enabled) {
				updateEnabledness(!enabled);
			}
		});
		updateEnabledness(!isEnabled.getValue());
	}

	public IntegerModel getComparisons() {
		return comparisons;
	}

	public IntegerModel getErrors() {
		return errors;
	}
	
	public DoubleModel getElicitationCost() {
		return elicitationCost;
	}

	public BooleanModel getIsExperimental() {
		return isExperimental;
	}

	public BooleanModel getIsEnabled() {
		return isEnabled;
	}	

	public BooleanModel getIsFixedSeed() {
		return isFixedSeed;
	}

	public long getTotalBids() {
		return totalBids;
	}

	private void updateEnabledness(boolean lock) {
		comparisons.setLock(lock);
		errors.setLock(lock);
		elicitationCost.setLock(lock);
		isFixedSeed.setLock(lock);
		isExperimental.setLock(lock);
	}

}
