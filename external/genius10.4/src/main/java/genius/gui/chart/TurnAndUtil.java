package genius.gui.chart;

/**
 * Stores a tuple <Turn number, Utility>.
 * 
 * @author W.Pasman
 *
 */
public class TurnAndUtil {
	private Double turn;
	private Double util;

	public TurnAndUtil(Double turn, Double util) {
		this.turn = turn;
		this.util = util;
	}

	public Double getTurn() {
		return turn;
	}

	public Double getUtil() {
		return util;
	}

}
