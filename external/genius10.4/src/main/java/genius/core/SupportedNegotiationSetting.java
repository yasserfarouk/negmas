package genius.core;

import genius.core.utility.UTILITYSPACETYPE;

/**
 * Indicates what negotiation settings are supported by an agent.
 */
public class SupportedNegotiationSetting {
	private final static UTILITYSPACETYPE defaultSpace = UTILITYSPACETYPE.NONLINEAR;
	/**
	 * LINEAR means: only supports linear domains, NONLINEAR means it supports
	 * any utility space (both linear and non-linear)
	 */
	private UTILITYSPACETYPE utilityspaceType = defaultSpace;

	public SupportedNegotiationSetting() {
	}

	public static SupportedNegotiationSetting getLinearUtilitySpaceInstance() {
		SupportedNegotiationSetting s = new SupportedNegotiationSetting();
		s.setUtilityspaceType(UTILITYSPACETYPE.LINEAR);
		return s;
	}

	public static SupportedNegotiationSetting getDefault() {
		return new SupportedNegotiationSetting();
	}

	boolean supportsOnlyLinearUtilitySpaces() {
		return utilityspaceType == UTILITYSPACETYPE.LINEAR;
	}

	public UTILITYSPACETYPE getUtilityspaceType() {
		return utilityspaceType;
	}

	public void setUtilityspaceType(UTILITYSPACETYPE utilityspaceType) {
		this.utilityspaceType = utilityspaceType;
	}

	/**
	 * returns human readible version. Bit hacky, I suspect this will change
	 * when we get more nonlinear agents.
	 */
	public String toExplainingString() {
		if (utilityspaceType == UTILITYSPACETYPE.NONLINEAR) {
			return "compatible with non-linear utility spaces";
		}
		return "";
	}

}
