package genius.core.boaframework;

/**
 * The type of BOA components
 */
public enum BoaType {
	BIDDINGSTRATEGY(OfferingStrategy.class), ACCEPTANCESTRATEGY(AcceptanceStrategy.class), OPPONENTMODEL(
			OpponentModel.class), OMSTRATEGY(OMStrategy.class),
	/** used if we could not determine the type. */
	UNKNOWN(null);

	private Class<? extends BOA> realclass;

	BoaType(Class<? extends BOA> realclass) {
		this.realclass = realclass;
	}

	/**
	 * @return
	 * @return the real root class type for this type.
	 */
	public Class<? extends BOA> getRealClass() {
		return realclass;
	}

	/**
	 * @param instance
	 * @return the type of the instance. UNKNOWN if not a known BOA class.
	 */
	public static BoaType typeOf(Class<? extends BOA> instanceClass) {
		BoaType[] vals = values();
		for (BoaType knowntype : vals) {
			Class<? extends BOA> knownclass = knowntype.getRealClass();
			if (knownclass != null && knownclass.isAssignableFrom(instanceClass)) {
				return knowntype;
			}
		}
		return UNKNOWN;
	}
}