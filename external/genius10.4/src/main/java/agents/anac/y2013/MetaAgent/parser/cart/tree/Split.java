package agents.anac.y2013.MetaAgent.parser.cart.tree;
import java.util.HashMap;

public class Split {

	public enum Direction{
		LEFT,RIGHT
	}

	protected String _name;
	protected double _value;
	protected Direction _direction;
	
	protected Split(String _name, double _value, Direction _direction) {
		this._name = _name;
		this._value = _value;
		this._direction = _direction;
	}

	public String getName() {
		return this._name;
	}

	public Direction getDirection(HashMap<String, Double> values) {
		double value=values.get(this._name);
		if(value<this._value)
			return this._direction;
		if(this._direction==Direction.LEFT)
			return Direction.RIGHT;
		return Direction.LEFT;
	}

}