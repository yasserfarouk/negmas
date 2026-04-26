package agents.anac.y2013.MetaAgent.parser.cart.tree;


public class PrimarySplit extends Split{
	double _improve;
	int _missing;
	

private PrimarySplit(String name, double value, Direction direction,
			double improve, int missing) {
		super(name,value,direction);
		this._improve = improve;
		this._missing = missing;
	}


public static PrimarySplit factory(String text){
	text=text.trim();
	String[]wordstext=text.split(" +");
	String name=wordstext[0];
	double value=Double.parseDouble(wordstext[2]);
	Direction dir;
	if(wordstext[5].contains("left"))
		dir=Direction.LEFT;
	else
		dir=Direction.RIGHT;
	double improve=Double.parseDouble(Node.substring(text,"improve=",","));
	int missing=Integer.parseInt(Node.substring(text, "(",text.indexOf("missing")).trim());
	PrimarySplit s=new PrimarySplit(name,value,dir,improve,missing);
	return s;
}


@Override
public String toString() {
	return "Split [_name=" + _name + ", _value=" + _value + ", _direction="
			+ _direction + ", _improve=" + _improve + ", _missing=" + _missing
			+ "]";
}
}