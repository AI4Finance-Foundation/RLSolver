import cirq
import numpy as np

QUBIT_ORDER = [
    cirq.GridQubit(0, 5),
    cirq.GridQubit(0, 6),
    cirq.GridQubit(1, 4),
    cirq.GridQubit(1, 5),
    cirq.GridQubit(1, 6),
    cirq.GridQubit(1, 7),
    cirq.GridQubit(2, 4),
    cirq.GridQubit(2, 5),
    cirq.GridQubit(2, 6),
    cirq.GridQubit(2, 7),
    cirq.GridQubit(2, 8),
    cirq.GridQubit(3, 2),
    cirq.GridQubit(3, 3),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(3, 7),
    cirq.GridQubit(3, 8),
    cirq.GridQubit(3, 9),
    cirq.GridQubit(4, 1),
    cirq.GridQubit(4, 2),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(4, 7),
    cirq.GridQubit(4, 8),
    cirq.GridQubit(4, 9),
    cirq.GridQubit(5, 0),
    cirq.GridQubit(5, 1),
    cirq.GridQubit(5, 2),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(5, 6),
    cirq.GridQubit(5, 7),
    cirq.GridQubit(5, 8),
    cirq.GridQubit(6, 1),
    cirq.GridQubit(6, 2),
    cirq.GridQubit(6, 3),
    cirq.GridQubit(6, 4),
    cirq.GridQubit(6, 5),
    cirq.GridQubit(6, 6),
    cirq.GridQubit(6, 7),
    cirq.GridQubit(7, 2),
    cirq.GridQubit(7, 3),
    cirq.GridQubit(7, 4),
    cirq.GridQubit(7, 5),
    cirq.GridQubit(7, 6),
    cirq.GridQubit(8, 3),
    cirq.GridQubit(8, 4),
    cirq.GridQubit(8, 5),
    cirq.GridQubit(9, 4),
]

CIRCUIT = cirq.Circuit(
    [
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.3173253018121125).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-1.6477173004723893).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.264041189400079).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.26280624622781834).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-2.5762285652409016).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.414163977160463).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.7690637873885737).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.3774012068579582).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.0114605673421808).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=1.138085630539364).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.7879539993524038).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=1.08307690004197).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.33966298910159587).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.04443220800534145).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.2147616940453956).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.163759918940749).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-2.85747020078256).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.7461541741686193).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.4835296607681667).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.6617633311972435).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.09928529861854263).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.1939260277947599).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.8139140883178477).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.915984339061019).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=2.4168261409239378).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=-2.2832984989084952).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.7246952848310728).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.916611500778383).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.0707940965761396).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.3010518408837803).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.9565969656162197).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-0.4644787859414882).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.2292952908456423).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.7705921356188823).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.7049240543353248).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.8540572282733239).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.0912658513256037).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.89847199517597).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-2.969109372876216).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-3.064995414802359).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=3.018621181244594).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.8951022794421073).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-0.5990585158397241).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=0.6433360120378406).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5553516266448077, phi=0.47754343999179444).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5513695622863453, phi=0.5525497751179018).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5920107910314218, phi=0.495277170657118).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.4850359189030258, phi=0.5432026764541129).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5586373808029026, phi=0.46099332137061977).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.5468483989685475, phi=0.5246739758652749).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5358532152984938, phi=0.5376855016240104).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.4939596076472883, phi=0.6686288484482796).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.548255653119813, phi=0.48307337635991665).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.6000940826263392, phi=0.5132890545539279).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5314832643812344, phi=0.47518174639867017).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5608608535579012, phi=0.6668893914368428).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5676963775554011, phi=0.48299203630424015).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.4807002448035869, phi=0.48033134534938904).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5479898631344362, phi=0.5021140664341368).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5508083238882544, phi=0.5088503466820911).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.507155881326315, phi=0.4773896323482684).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5602578123868405, phi=0.5276499696880497).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.3390062722902263, phi=0.4491086448159195).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.5485996563720543, phi=0.4873901665634375).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5468701891071555, phi=0.47909090411139443).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5253448615592025, phi=0.5174324533286649).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.16307624051571898).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-0.493468239175996).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.704941879610401).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.1780944439825038).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-2.6059724790645484).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.44390789098411).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.9298967345251388).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.5382341539945223).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.1517033926271893).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.025078329430006).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=2.542696554175027).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.2475736534854605).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.4826654696648385).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.777896250761093).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.2003466724171403).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.2513484475217864).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-3.0988342141692855).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.9875181875553447).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.2746469892271925).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.0964133187981153).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.204887498182803).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.9116761717694999).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=3.1306040112011395).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-1.5773171314004193).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-0.4763204339142323).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=0.6098480759296746).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.448253840739354).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.256337624792044).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.19061250940812013).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.03964523489952067).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.315564396581424).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.8234462169066923).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.0617504173025294).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.5204535725292896).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.2924761319403979).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.441609305878397).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.3434238870947954).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.646313959406778).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.8121295131057478).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=1.06121003260676).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-0.42560542409947544).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=0.5491243259019624).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-0.0834666499441532).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=0.1277441461422697).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.35735640822256265).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-0.1983553139014198).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=1.0739111967585693).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.7973576774008713).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.5737031091561227).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.5187574245887685).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.2201355249741619).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.6576474818092608).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.7055627757744753).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.507232552408385).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.0417252337038576).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.0538203123376384).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.970262116438331).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.8836250759129847).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.3098629297984665).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=2.443561195189549).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.4247246639327438).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.6059672967639296).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.8746786053740525).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.298546430144768).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.4789428742185589).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.3690259845995545).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.871758537992207).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.9176104314017053).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.235171088952629).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.9048441890146561).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.6852909058508807).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-0.30725407338960053).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.39365516427147).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.474901705533833).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.9654285295789933).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-2.244827484845411).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.1576388084312867).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.2296835627297327).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-2.462559713555187).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.473451472121912).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.198205529968215).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.7448164577820844).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.379234079589132).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=1.0076831864609233).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=2.9970389123402397).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-3.0895079172731412).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5390237928795745, phi=0.5165270526693762).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.6504170890091214, phi=0.5364505757789483).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.568670761625143, phi=0.5437849784272266).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5465254433686004, phi=0.5557344191201743).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.2913067233725621, phi=0.47167387649514164).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5381480272852108, phi=0.6090669301824796).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5830570458423034, phi=0.5523056687078169).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.546350568481482, phi=0.4879631369282715).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.5326394193330342, phi=0.4395048201320801).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5745817206538912, phi=0.5078468553250708).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5287959808590572, phi=0.5477757594186843).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4855370661448744, phi=0.5162805489497694).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.527373036078513, phi=0.5115160136560014).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4411810311735318, phi=0.4660991447993951).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.4000192936015627, phi=0.45553928758281914).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5800842999687046, phi=0.45282693980374283).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.518789513006022, phi=0.48179520284759936).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.529911511830527, phi=0.5594440380846348).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.4246757982684226, phi=0.4944809315999467).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.5296678467857068, phi=0.5740197418485297).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.4984631547270855, phi=0.49206955088399174).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.7763958689955128).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=0.9353969633166557).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=0.36274780954765795).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.08619429018995639).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-0.3018338960579392).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.2468882114905817).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.911310617996854).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=0.4737986611617542).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-1.0836255652629516).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.281955788629042).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.5542903568833175).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.566385435517102).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-3.0244571487869223).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.937820108261576).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.7779233629240174).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-0.6442250975329351).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.9832730401849048).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.802030407353719).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.4439575092147923).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.13217466601449424).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.403696470986718).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.2937795813677138).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.21607554092298906).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=0.2619274343324891).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=1.1779792840846168).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.8476523841466443).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.2984429622845646).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-0.6941020169559167).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-0.9593104645965189).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.0405570058588802).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=0.8625137615366966).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-0.14191271680311424).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.7589110211533026).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.8309557754517485).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-1.3760940120951801).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=1.3869857706619086).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.379437509198123).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.71960082862741).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.7698739356164876).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-3.1414248287446966).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=0.23785197077916556).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-0.3303209757120671).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.77942643322171).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-2.9693179420992077).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.2922430309899244).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-1.6783000321440404).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-0.6593126012915604).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=0.7608655182744215).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.023721273541816856).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.03855239873136185).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.286031015354709).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.77499613380445).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.662151913997727).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-0.6582829725639332).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.713557275735157).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=2.744437270934066).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.202984923444255).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=2.1701292716579985).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.019423349111488).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.0502115478476775).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.3112987566698795).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.0934311692598992).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.7679657389508385).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.715098043895569).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.7808733393854581).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.31994018625179876).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.3758452824097045).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.0377942457069107).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.728424385842244).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.637078948735807).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=0.17120944395221738).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.05599741835860872).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.0251633459989407).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-1.980388483241505).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=1.3214782847709756).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.632633471925928).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.9248895619515345).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.8864865803847195).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.5623387365712311).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.2948437115959983).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.862667395282668).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.541720335197579).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=2.7548237094799237).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-2.579629919264823).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4660264742068387, phi=0.4703035018096913).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5595675891930325, phi=0.5029212805853078).on(
                cirq.GridQubit(0, 6), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5057896135518938, phi=0.5276098280644357).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.544885207693353, phi=0.5728958894277337).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.521825393645828, phi=0.5386243364775084).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.568786150985278, phi=0.5431362030835812).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.6068421332516425, phi=0.49425168255721214).on(
                cirq.GridQubit(2, 8), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.5926635714606605, phi=0.49518543009178106).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.4823586616735414, phi=0.52880608763343).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.5878491175464968, phi=0.4879472611318097).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5390712580125911, phi=0.5384288567835004).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5319407509110305, phi=0.5000474088191262).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5700853308915463, phi=0.46400287842866444).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5356042072853593, phi=0.5206952948829173).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.6447139280388123, phi=0.46839573172584353).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5383311289896362, phi=0.5127577213621436).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
            ),
            cirq.FSimGate(theta=1.5052207980661314, phi=0.5125556567248526).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.3373234810096835, phi=0.5739054404737827).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.5012621338159504, phi=0.5119166112294746).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.3909778451077717, phi=0.4282244069641004).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5178422515660452, phi=0.47308231008168455).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.9861770190216479).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-1.1760685278991456).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.2885842536844585).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=0.9025272525303425).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=2.812795931538986).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-2.711243014556125).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.164433610947057).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.1496024857575122).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.20531470934292176).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.28365040910681927).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.7012188350729307).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.3807839485112705).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.4684910795794703).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.499371074778379).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=3.0033298494397513).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-3.036185501226008).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.457591862048062).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.488380060784255).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.969159884330006).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.7512922969200257).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.2167886285113614).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.1639209334560885).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.541356421261149).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.0804232681274897).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.3146808118107813).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.9010412836941626).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=3.0416040572865324).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-3.132949494392969).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.843102255783201).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.7278902301895904).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=0.023008817691426486).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.0217660450660091).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-2.6776476581410904).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=2.366492470986138).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.5691988828353551).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-3.0410080476786856).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.4084288060235757).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.6759238309988085).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.5383294814030952).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.217382421318006).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-1.9159628617918134).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=2.091156652006914).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.8973785970103734).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=1.023846054779412).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-2.3821944378581517).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.4157551756470657).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.193103489734927).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-0.863824430878733).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-3.0033656835588616).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=1.5488394013159095).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.4261379546591897).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.301572450843558).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.240550278619075).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-2.4994636160960297).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.5795184523373251).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.336490509367934).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.012209678462856033).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.21149726443332995).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.6140089890307046).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.581194025726507).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-3.006664670625252).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.923675144087671).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.4054198972849292).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.38285099061872785).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.083735472947197).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=3.004712952751067).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-1.184431491824756).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=0.35249667437291077).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-2.066891175965722).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.0471485529289026).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.7069919724503997).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.74476637272657).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.43519181041457244).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.22274950137185104).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-0.21318465204709372).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.3264964935359522).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.4960410664021744).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.3023575889455294).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-1.6421952570917462).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.2615660964943665).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=1.1368196073533916).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.131323125840769).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=2.0989351150540436).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-1.6300878062136483).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-1.4262853050657043).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=1.4418485627396933).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5496057658133178, phi=0.5393238727561925).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(2, 4)
            ),
            cirq.FSimGate(theta=1.5258573993033773, phi=0.5301963978982887).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.5999105816064956, phi=0.47191843749859924).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.4104416482963105, phi=0.5491205037550326).on(
                cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5319225163456216, phi=0.49105711881659464).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.465404631262162, phi=0.4878297652759932).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.4707183263040744, phi=0.5651156081809627).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5467222527282019, phi=0.5068034228124387).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5496604814896044, phi=0.5531983372087332).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5524939155045483, phi=0.5566693071275687).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.363213412756342, phi=0.7994150767548583).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.5284460090316794, phi=0.5432682295494672).on(
                cirq.GridQubit(3, 9), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.3166607142943418, phi=0.40127686655135986).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)
            ),
            cirq.FSimGate(theta=1.5594498457697554, phi=0.5290555375763617).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.5125120660103082, phi=0.5195553885293336).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.54223977200166, phi=0.47914474770071624).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.6074361272506783, phi=0.4906717148713765).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.4382571284509336, phi=0.5684944786275166).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.533964395594986, phi=0.47457535225216274).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.5001433000839564, phi=0.6605028795204214).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)
            ),
            cirq.FSimGate(theta=1.640026880988053, phi=0.4976366432039147).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)
            ),
            cirq.FSimGate(theta=1.5423122824843978, phi=0.7792646973911541).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.4034683130634775).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-1.277000855294439).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=0.8094742909749842).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.7759135531860721).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.4960244225277073).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.8253034813839015).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.5332984250194883).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-1.9878247072624404).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-0.046993842598290314).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=0.1715593464139289).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.476043839698896).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.2171305022219414).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.2489925740650705).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.0059646310956794).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.873626602871628).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.072914188842102).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.7309413876467765).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.763756350950974).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.8139707966637282).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-0.8969603232013093).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.06548253516807989).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.08805144183428126).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.15068949854421732).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=1.0716669783480874).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-1.2986720964769356).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=0.4667372790250903).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-0.9305306035324108).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.9107879804955914).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-0.4460931211417112).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.4838675214178778).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=2.630217089772671).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.8426593988153925).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.6879809275483986).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.57466908605954).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.4245743657884624).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.38174215675489265).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.455505212825784).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.8361343734231639).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-3.0126798754623003).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=3.018176356974923).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=0.9504893612700336).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.4816420524296383).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-1.9405812002483884).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=1.9561444579223775).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.794847385157855).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-2.12523938381813).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.3821533610810093).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.909000796708909).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-2.1238392231237526).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.9617746350433123).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-0.07819269292208375).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.4698552734526977).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.2250888677864609).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=1.3517139309836423).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.9892139064207282).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.6940910057311618).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.2939651907647836).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.589195971861038).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.6861228455545643).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.737124620659209).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-2.37994811743682).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.2686320908228783).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.722290702441038).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.9005243728701142).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.8431924974582312).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.136403823871534).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.5036077538729202).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.04967912592778844).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=1.0093926321157198).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=-0.8758649901002791).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.7687557459606715).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.576839530013361).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.2665508376949681).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.0362930933873287).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-0.212075501519128).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.7041936811938596).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.907879304020952).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.4491761487941943).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.7005154892535652).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.849648663191562).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.44507130084451413).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-1.7385187615225244).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=3.100447633859094).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-2.85136711435808).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.6183695754539045).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.7418884772563887).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.9615361913394267).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=3.0058136875375467).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5553516266448077, phi=0.47754343999179444).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5513695622863453, phi=0.5525497751179018).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5920107910314218, phi=0.495277170657118).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.4850359189030258, phi=0.5432026764541129).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5586373808029026, phi=0.46099332137061977).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.5468483989685475, phi=0.5246739758652749).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5358532152984938, phi=0.5376855016240104).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.4939596076472883, phi=0.6686288484482796).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.548255653119813, phi=0.48307337635991665).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.6000940826263392, phi=0.5132890545539279).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5314832643812344, phi=0.47518174639867017).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5608608535579012, phi=0.6668893914368428).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5676963775554011, phi=0.48299203630424015).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.4807002448035869, phi=0.48033134534938904).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5479898631344362, phi=0.5021140664341368).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5508083238882544, phi=0.5088503466820911).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.507155881326315, phi=0.4773896323482684).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5602578123868405, phi=0.5276499696880497).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.3390062722902263, phi=0.4491086448159195).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.5485996563720543, phi=0.4873901665634375).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5468701891071555, phi=0.47909090411139443).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5253448615592025, phi=0.5174324533286649).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.31444584283002364).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-0.015946155830254938).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.932048877088093).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.8242889944635934).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-3.0583618211816965).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.8962972331012597).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.5060320923437907).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.8976946728744064).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.365331693071468).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.2387066298742866).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.23447135159810628).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.5295942522876693).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.8490372897984599).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.1442680708947144).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.5826474399967676).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.6336492151014124).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=2.706829009664558).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.8181450362784997).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.5134080309000622).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.3351743604709858).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-3.135820012920009).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.8541539678462762).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.18691783098963555).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.7402047107903442).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=0.9311130748939824).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=-0.7975854328785417).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.0451971900523898).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=1.2371134059997004).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.1467324248629893).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.3769901691706288).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.798948443462815).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.9921186840420395).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.7403344304778443).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=3.0841477214749844).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.29688469702216125).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.44601787096015855).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.2935668696036977).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.0001194089256877).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.5985012126614713).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=0.8475817321624852).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-2.071799974580564).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.1953188763830482).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.2790110255555547).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-2.2347335293574346).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.834878491568304).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-0.675877397247163).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.381840722190283).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.658394241547988).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.9308711355071537).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.8759254509397962).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.8987195381494715).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.336231494984574).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-1.3354989129001176).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.5338291362662098).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.71561957998977).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.703524501355993).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.9630118858562682).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.0496489263816144).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.1160577214341174).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=1.2497559868251926).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-3.08129273747371).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.900050104642524).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.9058518772312).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.8012012547190963).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.0391199027161164).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.9292030130971156).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.6342588634142459).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.5884069700047476).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.926321472742236).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.5959945728042637).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.627768701928069).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.6352237226875843).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.004574550754157).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.9233280094917975).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=0.6280835953078849).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=0.09251744942569928).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-1.3402207874422203).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=1.2681760331437744).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.9985018545423403).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.9876100959756118).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-1.7727675841692694).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.43260426473998237).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.316402226517212).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.9448513333890034).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=2.0545611162634643).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-2.147030121196366).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5390237928795745, phi=0.5165270526693762).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.6504170890091214, phi=0.5364505757789483).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.568670761625143, phi=0.5437849784272266).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5465254433686004, phi=0.5557344191201743).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.2913067233725621, phi=0.47167387649514164).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5381480272852108, phi=0.6090669301824796).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5830570458423034, phi=0.5523056687078169).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.546350568481482, phi=0.4879631369282715).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.5326394193330342, phi=0.4395048201320801).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5745817206538912, phi=0.5078468553250708).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5287959808590572, phi=0.5477757594186843).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4855370661448744, phi=0.5162805489497694).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.527373036078513, phi=0.5115160136560014).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4411810311735318, phi=0.4660991447993951).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.4000192936015627, phi=0.45553928758281914).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5800842999687046, phi=0.45282693980374283).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.518789513006022, phi=0.48179520284759936).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.529911511830527, phi=0.5594440380846348).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.4246757982684226, phi=0.4944809315999467).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.5296678467857068, phi=0.5740197418485297).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.4984631547270855, phi=0.49206955088399174).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.2539179523412542).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=1.4129190466623953).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.464685578683067).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.741239098040772).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.0553341302930903).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.1102798148604478).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-2.5898946311721645).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.152382674337062).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.957436123411643).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.7591059000455507).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.02844986339736266).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.0405449420311399).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.6745458439019316).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=0.5879088033765854).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.41588184544033885).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=0.549580110831414).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=1.522744361221548).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.7039869940527337).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.05869731535953804).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.6348294905888281).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.8435194424891534).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.7336025528701526).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.561092364850147).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-2.5152404714406487).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.486828900295011).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.15650200035703854).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.6440348337926238).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.6365798130331086).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.9256451275574413).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-0.8443985862950818).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-3.083326611371781).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-2.479257651074221).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-0.02641469015277664).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.04563006414566928).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.44602972698687887).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.4351379684201504).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-1.932774683843977).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.5926113644146902).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.707042082544561).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-3.078592975672777).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=1.180329766855941).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-1.2727987717888425).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.2033204572781457).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-0.39321196615563997).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.0660483599313508).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-1.4521053610854686).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.7902859565836164).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.8918388735664704).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=1.1700839348221166).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.1552528096325716).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.31116375658371).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.800128875033451).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.8438654874083156).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.118884933209614).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.2360351923894157).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=2.2669151875883244).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.258076644653265).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-2.2909322964395287).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.5095118030715255).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.540300001807715).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.4658691491036641).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.683736736513648).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-3.0952196888062318).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-3.135097923318085).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.9837281957733806).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.444661348907033).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.4492964337789402).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.0356569056623215).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-0.8811679055315872).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.7898224684251467).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-0.30631263939352493).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.4215246649871318).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.2262252758289165).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.181450413071481).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=1.9874959273319848).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.298651114486937).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.117793686248575).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.6935824560876753).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.272351373125133).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.5398463981003658).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.8569783065676502).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-1.1779253666527394).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-2.1460608301200352).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=2.3212546203351394).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4660264742068387, phi=0.4703035018096913).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5595675891930325, phi=0.5029212805853078).on(
                cirq.GridQubit(0, 6), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5057896135518938, phi=0.5276098280644357).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.544885207693353, phi=0.5728958894277337).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.521825393645828, phi=0.5386243364775084).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.568786150985278, phi=0.5431362030835812).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.6068421332516425, phi=0.49425168255721214).on(
                cirq.GridQubit(2, 8), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.5926635714606605, phi=0.49518543009178106).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.4823586616735414, phi=0.52880608763343).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.5878491175464968, phi=0.4879472611318097).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5390712580125911, phi=0.5384288567835004).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5319407509110305, phi=0.5000474088191262).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5700853308915463, phi=0.46400287842866444).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5356042072853593, phi=0.5206952948829173).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.6447139280388123, phi=0.46839573172584353).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5383311289896362, phi=0.5127577213621436).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
            ),
            cirq.FSimGate(theta=1.5052207980661314, phi=0.5125556567248526).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.3373234810096835, phi=0.5739054404737827).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.5012621338159504, phi=0.5119166112294746).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.3909778451077717, phi=0.4282244069641004).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5178422515660452, phi=0.47308231008168455).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.7209023122143705).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=2.5310108033368763).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.0623895826258867).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=0.6763325814717689).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.3394160203485583).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.4409689373314123).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.9706284025831167).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-0.9557972773935718).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.23044745057192983).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.25851766787781116).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.07594907070061652).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.3963839572622732).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.9460131629252118).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.9768931581241205).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.4577317186577758).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=1.4248760668715121).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=1.967503408088021).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.9982916068242105).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.536857517076033).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.7547251044860168).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.20321125091115985).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.2560789459664292).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.9772273507595983).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.4381605038932506).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.2412296604415545).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.1724098676750643).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=1.1943475769758756).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.285693014082316).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.9625609680506457).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=3.0777729936442526).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-0.17805311213854935).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.22282797489598494).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=2.939520006477494).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=3.03251011354714).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.3762947585383145).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-2.8481039233816503).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=0.5737386963272115).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.8412337213024443).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.025210123926172834).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.34615718401126205).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=2.9849216778081527).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-2.8097278875930485).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.5633962395713752).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=1.6898636973404277).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-3.0733448216477566).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=3.1069055594366706).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.6005369985431415).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.2712579396869472).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.7969941045801787).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=0.34246782233724815).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-0.4012954335715548).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=0.5258609373871863).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.10426727417833348).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.36318061165530224).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.0217202648352455).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.2647482078046224).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.1313300474439032).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.9320424614734223).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.5580260549363416).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.590841018240546).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.4107356026013562).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.327746076063768).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.4921020654833157).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=1.5146709721495242).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.7185651740549375).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=0.2024123057489291).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=0.9015860301591854).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.7335208476110304).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=0.4589493175202506).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.47869194055706993).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.9299987842480917).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.8922243839719286).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.012788966620810527).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.22523127566352488).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-2.098140244201055).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.2114520856899205).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.0225427056185836).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.2162261830752286).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.398866431583265).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-0.7794955921806448).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=0.23204092311950575).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-0.22654444160688317).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-2.060533558298708).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.5293808671391034).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=2.9719444099599244).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.9563811522859353).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5496057658133178, phi=0.5393238727561925).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(2, 4)
            ),
            cirq.FSimGate(theta=1.5258573993033773, phi=0.5301963978982887).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.5999105816064956, phi=0.47191843749859924).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.4104416482963105, phi=0.5491205037550326).on(
                cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5319225163456216, phi=0.49105711881659464).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.465404631262162, phi=0.4878297652759932).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.4707183263040744, phi=0.5651156081809627).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5467222527282019, phi=0.5068034228124387).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5496604814896044, phi=0.5531983372087332).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5524939155045483, phi=0.5566693071275687).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.363213412756342, phi=0.7994150767548583).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.5284460090316794, phi=0.5432682295494672).on(
                cirq.GridQubit(3, 9), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.3166607142943418, phi=0.40127686655135986).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)
            ),
            cirq.FSimGate(theta=1.5594498457697554, phi=0.5290555375763617).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.5125120660103082, phi=0.5195553885293336).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.54223977200166, phi=0.47914474770071624).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.6074361272506783, phi=0.4906717148713765).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.4382571284509336, phi=0.5684944786275166).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.533964395594986, phi=0.47457535225216274).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.5001433000839564, phi=0.6605028795204214).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)
            ),
            cirq.FSimGate(theta=1.640026880988053, phi=0.4976366432039147).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)
            ),
            cirq.FSimGate(theta=1.5423122824843978, phi=0.7792646973911541).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.0694859556244936).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-1.9430184978554552).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.5006246747645928).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.4670639369756788).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.9034579313359217).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-3.0504483169874703).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.673073153959173).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-0.7814531282837649).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.78043954563244).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.6558740418168085).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.6602391647418315).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.9191525022188002).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.4329540159419452).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=1.675981958911322).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-3.01716632877838).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-3.066731392430725).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.441093656320284).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.4082786930160793).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.7819582713601747).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=0.6989687448225865).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.8320394276001863).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.8094705209339779).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.9529901455463587).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-2.409217681829361).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=2.898495688718709).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.5527548010090317).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=2.82681421016121).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.8465568331980293).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.2001014293393801).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.16232702906321).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=2.1822363127372952).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.3946786217800096).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-1.7102487874772194).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=1.8235606289660848).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-3.0940094062322956).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.3828593784039356).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.5855564758492235).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.2049273152518438).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-2.1079011912284145).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.113397672741037).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-1.1732272725567938).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.642074581397189).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-0.05562560809442374).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.07118886576841277).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 0)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.2723694685035944).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-2.6027614671638695).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.0283479115620935).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.7279899599895927).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-1.671449881006609).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.5093852929261686).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.9254491732327352).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-2.96607355341623).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.4387171682307383).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=1.5653422314279197).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.516803494985723).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=1.811926395675286).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.927593370631163).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=3.060361155452169).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.3038220779749388).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.3548238530795835).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-1.9024260340910786).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=1.791110007477137).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.9610517441139077).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.139285414542986).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.7856702935350057).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=3.078881619948305).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.4620557111158377).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=3.0153425909165463).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-0.39804087669249455).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=0.5315685187079389).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.020978530427171016).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.2128947463744817).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.6792895352135098).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.9095472795211528).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.3807479686544752).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=1.872866148329205).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.696721989983331).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=3.1277601619694977).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.6961069241718005).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.845240098109798).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.2011232496365736).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.0923242110414364).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=2.886819333414813).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-2.6377388139138063).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.027824975027186838).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=0.0956939267752972).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.9591714403404623).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-0.9148939441423564).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5553516266448077, phi=0.47754343999179444).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5513695622863453, phi=0.5525497751179018).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5920107910314218, phi=0.495277170657118).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.4850359189030258, phi=0.5432026764541129).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5586373808029026, phi=0.46099332137061977).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.5468483989685475, phi=0.5246739758652749).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5358532152984938, phi=0.5376855016240104).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.4939596076472883, phi=0.6686288484482796).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.548255653119813, phi=0.48307337635991665).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.6000940826263392, phi=0.5132890545539279).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5314832643812344, phi=0.47518174639867017).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5608608535579012, phi=0.6668893914368428).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5676963775554011, phi=0.48299203630424015).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.4807002448035869, phi=0.48033134534938904).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5479898631344362, phi=0.5021140664341368).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5508083238882544, phi=0.5088503466820911).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.507155881326315, phi=0.4773896323482684).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5602578123868405, phi=0.5276499696880497).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.3390062722902263, phi=0.4491086448159195).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.5485996563720543, phi=0.4873901665634375).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5468701891071555, phi=0.47909090411139443).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5253448615592025, phi=0.5174324533286649).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.7919679261757615).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=0.4615759275154865).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.2858543266070086).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.8127017622349089).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=2.7724341438807443).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-2.9344987319611846).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.34122438796686083).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.0504381925637531).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.5789599935157455).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.452334930318564).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-3.011639257371243).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.97642314911878).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.7845908900679195).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.4893601089716651).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.3182370996031914).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.2672353244985466).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=2.22930692631882).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.340622952932762).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.7521690725729311).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.5739354021438547).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.1933422168432344).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.7966317639230525).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.7787456339991223).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-1.2254587541984137).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=2.3385465837022004).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=-2.205018941686756).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=1.7445370863354517).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-1.5526208703881421).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.7991079480454923).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.5688502037378491).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.6302759763274668).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.1223941560021977).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.8642668635264386).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.405563708299681).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.3012932621039184).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.45042643604191573).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.6473723191226171).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.6460751415553929).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.38487291221719744).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=0.6339534317182043).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.5651907821179307).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.4416718803154467).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.6416966061243414).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=1.6859741023224473).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.3124005749140437).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-1.1533994805929026).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=0.44559266604046144).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.16903914668277054).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.995146145321435).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.0500918298887782).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.577303551324789).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-3.0148155081598844).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.9066247056048697).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.7082944822387844).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.18977908650382602).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.17768400787003458).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.3868994190287225).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.3002623785033762).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.07774748693024236).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=0.055950778460832844).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-0.3041248317005767).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.12288219886939088).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.4031970526568642).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.97932922788614).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.5992969312136793).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.4893800415946785).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.1429090423588804).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.1887609357683857).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.665713450647747).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.9960403505857194).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.570246498005254).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=1.5777015187647692).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.11961895860019921).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-0.03837241733783969).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-1.7092613389632234).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=2.4298623836968076).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.445104923863859).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.5171496781623048).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.17637811546030235).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.16548635689358804).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.5394446088728344).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.8796079283021287).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.253570373445214).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.8820194803170338).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=1.1120833201866915).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-1.204552325119593).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5390237928795745, phi=0.5165270526693762).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.6504170890091214, phi=0.5364505757789483).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.568670761625143, phi=0.5437849784272266).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5465254433686004, phi=0.5557344191201743).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.2913067233725621, phi=0.47167387649514164).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5381480272852108, phi=0.6090669301824796).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5830570458423034, phi=0.5523056687078169).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.546350568481482, phi=0.4879631369282715).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.5326394193330342, phi=0.4395048201320801).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5745817206538912, phi=0.5078468553250708).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5287959808590572, phi=0.5477757594186843).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4855370661448744, phi=0.5162805489497694).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.527373036078513, phi=0.5115160136560014).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4411810311735318, phi=0.4660991447993951).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.4000192936015627, phi=0.45553928758281914).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5800842999687046, phi=0.45282693980374283).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.518789513006022, phi=0.48179520284759936).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.529911511830527, phi=0.5594440380846348).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.4246757982684226, phi=0.4944809315999467).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.5296678467857068, phi=0.5740197418485297).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.4984631547270855, phi=0.49206955088399174).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.731440035686994).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=1.890441130008135).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=0.9910663402657676).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.7145128209080767).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=2.412502156644095).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.4674478412114382).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=3.014706662832104).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.830966687512387).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-1.2846874950933511).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.4830177184594362).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.4973906300885886).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.485295551454797).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.6753654609830588).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.7620025015084053).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.6096870538046986).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=1.7433853191957736).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.2544235445515852).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.0731809117203994).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.561352139933895).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.1374843151631708).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.9998428931879957).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-3.1097597828069965).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.9449250365563131).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=0.9907769299658185).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.20432148349459212).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.5346483834325646).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.5865126298698085).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.5790576091102935).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.8106007197114025).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.729354178449043).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-0.7459816771006729).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=1.466582721834257).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.811740401458856).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=2.73969564716041).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.2681534660689096).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.2572617075021952).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.038198430293498).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.904823557456794).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.6442102294726197).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-3.015761122600807).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=2.1228075629327137).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-2.215276567865615).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.3727855186654168).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=2.1828940097879226).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=0.839853688872779).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-1.2259106900268932).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.9212593118756303).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=3.022812228858484).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.36388914318605).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.349058017996505).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.336296497812718).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.8252616162624733).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.06669758163477013).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.3871324681964268).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.758513109043669).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.7893931042425848).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.4359529055712486).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-0.4688085573575122).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.999600257031581).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-3.0303884557677705).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=3.04014825230238).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.8222806648923964).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.6752198093837052).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.728087504438989).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.5348555762473737).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.0739224231137214).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.760437722031412).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-3.1091080570315555).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.9660885747790716).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.057434011885512).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-0.7838347227392681).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.8990467483328751).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.4272872056588852).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.3825123429014496).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=2.6535135698929793).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.9646687570479315).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.310697810545591).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.5006783317906596).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=3.1070414828214865).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.908648799382867).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.706561298761617).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.3856142386765278).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-0.7637600625404133).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=0.9389538527555175).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4660264742068387, phi=0.4703035018096913).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5595675891930325, phi=0.5029212805853078).on(
                cirq.GridQubit(0, 6), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5057896135518938, phi=0.5276098280644357).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.544885207693353, phi=0.5728958894277337).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.521825393645828, phi=0.5386243364775084).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.568786150985278, phi=0.5431362030835812).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.6068421332516425, phi=0.49425168255721214).on(
                cirq.GridQubit(2, 8), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.5926635714606605, phi=0.49518543009178106).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.4823586616735414, phi=0.52880608763343).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.5878491175464968, phi=0.4879472611318097).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5390712580125911, phi=0.5384288567835004).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5319407509110305, phi=0.5000474088191262).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5700853308915463, phi=0.46400287842866444).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5356042072853593, phi=0.5206952948829173).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.6447139280388123, phi=0.46839573172584353).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5383311289896362, phi=0.5127577213621436).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
            ),
            cirq.FSimGate(theta=1.5052207980661314, phi=0.5125556567248526).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.3373234810096835, phi=0.5739054404737827).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.5012621338159504, phi=0.5119166112294746).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.3909778451077717, phi=0.4282244069641004).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5178422515660452, phi=0.47308231008168455).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.144796336270808).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-0.04509517260668616).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.8361949115673113).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=0.450137910413197).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.2084426650565447).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.3099955820393987).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.2231768057808168).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.23800793097036174).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.25558019180095215).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.23338492664880306).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.853116976474162).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.1096334441437676).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.423535246270955).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=2.4544152414698637).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.36439202042424057).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-0.3972476722105043).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=1.4774149541279797).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.5082031528641693).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.240310388697509).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.022442801287525).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.6232111303336723).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.6760788253889418).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.7873741843992124).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.32644103126556).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.8322214909276797).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-3.037324288135288).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-0.652908903334783).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.5615634662283426).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.485038884704906).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.600250910298513).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-0.3791150419685252).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.4238899047259608).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=2.2735023639164993).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.5846575510714516).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.1833906342412845).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-2.6551997990846203).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-1.260951413369142).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.9934563883939022).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.58874972925544).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.909696789340529).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=1.6026209102285236).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-1.4274271200134194).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.229413882132363).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=2.3558813399014156).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.5186901017422194).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.485129363953309).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.2752147998282197).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=2.604493858684421).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.5906225256014963).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-0.8639037566414415).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=3.0544564853772584).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.929890981561627).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.0320157302624082).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.7731023927854395).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.6602263251717064).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.4171983822023293).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.274869773350659).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.075582187380178).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.5531242082761949).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.5203092449719904).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.185193465422536).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-0.26818299196011).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.893561278927976).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.8709923722617674).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.762319486122511).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-2.599888341253209).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=2.9876035521431206).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.4636469375846204).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=2.9847898110061664).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-3.0045324340429858).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.28380423376700037).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.24602983349083732).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.4607697436561793).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.6732120526989078).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.300089470824574).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.186777629335708).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.7420588295402624).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=1.5483753520836245).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-1.8432571869213135).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.4626280263239337).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.6727377611143694).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.6782342426270063).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=0.06318307552812286).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=0.4056642333122724).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=1.0869888178059526).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.0714255601319636).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5496057658133178, phi=0.5393238727561925).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(2, 4)
            ),
            cirq.FSimGate(theta=1.5258573993033773, phi=0.5301963978982887).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.5999105816064956, phi=0.47191843749859924).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.4104416482963105, phi=0.5491205037550326).on(
                cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5319225163456216, phi=0.49105711881659464).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.465404631262162, phi=0.4878297652759932).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.4707183263040744, phi=0.5651156081809627).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5467222527282019, phi=0.5068034228124387).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5496604814896044, phi=0.5531983372087332).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5524939155045483, phi=0.5566693071275687).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.363213412756342, phi=0.7994150767548583).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.5284460090316794, phi=0.5432682295494672).on(
                cirq.GridQubit(3, 9), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.3166607142943418, phi=0.40127686655135986).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)
            ),
            cirq.FSimGate(theta=1.5594498457697554, phi=0.5290555375763617).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.5125120660103082, phi=0.5195553885293336).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.54223977200166, phi=0.47914474770071624).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.6074361272506783, phi=0.4906717148713765).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.4382571284509336, phi=0.5684944786275166).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.533964395594986, phi=0.47457535225216274).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.5001433000839564, phi=0.6605028795204214).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)
            ),
            cirq.FSimGate(theta=1.640026880988053, phi=0.4976366432039147).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)
            ),
            cirq.FSimGate(theta=1.5423122824843978, phi=0.7792646973911541).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.7355035981855025).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-2.6090361404164497).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.191775058554196).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.1582143207652855).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.9722938670354466).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.6430148081792453).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.8794447329378625).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=0.42491845069492484).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-0.6753123733163804).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=0.7998778771320119).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.7965221691825732).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-3.055435506659542).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.1682847012306823).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.9252567582613054).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.1224792524944505).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.9231916665239694).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.32994339310774734).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.2971284298035428).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.37788733938406).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.294897812846486).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-2.5536239168111052).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.5761928234773137).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.5278945146310896).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=0.39308296517277697).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=0.8124781667347705).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.6444129841866155).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=0.30097371667529416).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.32071633971211355).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.846295979820468).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.808521579544305).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=1.7342555357019267).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.946697844744655).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.17470680467673816).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-0.06139496318787252).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.3294078710734496).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=1.1357243936168189).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.656567142655355).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.0371963032527347).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-1.2031225069945393).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.208618988507176).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=2.9862414007959472).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.517394091955552).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=1.8293299840595336).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.8137667263855448).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.7498915518493305).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-3.0802835505096056).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.6086428451364085).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.0817954095085085).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-1.2190605388894582).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.0569959508090179).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.510479653636196).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.1188170731055749).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.6523454686750088).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=1.7789705318722042).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.2603644107874103).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.9652415100978473).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.7219637566820456).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.4267329755857912).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-0.07847868960468674).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.02747691450003487).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-1.424903950745339).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=1.3135879241314043).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-3.0833725213928105).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.3780464562158556).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.7281480896117785).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.2618258911545084).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.8554661310749976).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.3021792512742607).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-1.805474385500716).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=1.939002027516153).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-2.8107128068150136).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=3.0026290227623225).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.341944600942405).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.572202345250048).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.54942043578982).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=3.041538615464553).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.0181379768080205).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.476841132034778).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.691698359090033).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.8408315330280303).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.847317800117658).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.5538703394396478).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=2.673191032970543).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-2.424110513469536).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.674019525508271).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-1.550500623705787).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.4033062351592847).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=1.4475837313573905).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5553516266448077, phi=0.47754343999179444).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5513695622863453, phi=0.5525497751179018).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5920107910314218, phi=0.495277170657118).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.4850359189030258, phi=0.5432026764541129).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5586373808029026, phi=0.46099332137061977).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.5468483989685475, phi=0.5246739758652749).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5358532152984938, phi=0.5376855016240104).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.4939596076472883, phi=0.6686288484482796).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.548255653119813, phi=0.48307337635991665).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.6000940826263392, phi=0.5132890545539279).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5314832643812344, phi=0.47518174639867017).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5608608535579012, phi=0.6668893914368428).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5676963775554011, phi=0.48299203630424015).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.4807002448035869, phi=0.48033134534938904).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5479898631344362, phi=0.5021140664341368).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5508083238882544, phi=0.5088503466820911).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.507155881326315, phi=0.4773896323482684).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5602578123868405, phi=0.5276499696880497).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.3390062722902263, phi=0.4491086448159195).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.5485996563720543, phi=0.4873901665634375).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5468701891071555, phi=0.47909090411139443).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5253448615592025, phi=0.5174324533286649).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.2694900095214976).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=0.9390980108612226).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.3603402238740756).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.16650721175382444).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=2.3200448017635935).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-2.482109389844034).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.188480868277516).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.7968182877468949).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.7925882939600157).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.6659632307628343).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=0.49437814403520974).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.19925524334564668).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.4182190699342954).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.1229882888380445).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-0.9359363320235802).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.8849345569189282).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=1.7517848429730876).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.8631008695870224).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.9909301142458062).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.8126964438167263).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.2508644207664616).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.544075747179761).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.5387762081916847).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.092063087992422).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-2.5372052146691644).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=2.6707328566846016).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.748913944456293).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=1.9408301604036016).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.5382369862256127).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.7684947305332557).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-0.46160350919211973).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.9537216888668532).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.185682850351128).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.7269796951243703).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.30570182718568617).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.45483500112368347).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.9988222313584671).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.292269692036477).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.17124461177292716).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=0.420325131273934).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.9189962316368465).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.7954773298343625).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.7207810693753984).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-0.6765035731772926).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.7899226582597905).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-1.6309215639386494).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-3.010159252908366).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.9964725349135293).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.6379781189704445).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-0.6929238035378162).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.2558875645000995).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.5897857858443913).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-1.134436983069726).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.3327672064358111).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.336061406982097).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.3481564856158883).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.546374583265859).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.633011623791205).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=1.2715526952946234).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-1.1378544299035482).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.4730430740725566).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.6542857069037424).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.9005422280824718).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.4766744033117476).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.15947395971125644).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.04955707009225563).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.3631083590475725).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.3172564656380672).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-1.9745630668581384).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.304889966796111).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.7704610130971474).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.520179314841961).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.7653366335537513).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.846583174816125).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.2365790339452616).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-1.5159779892116774).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.052754672009634).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=1.980709917711188).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-1.6457456236216925).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=1.6566373821884355).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.8516568019149346).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.0913651858353575).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.1907385203732446).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.8191876272450642).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=0.1696055241099188).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-0.26207452904281325).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5390237928795745, phi=0.5165270526693762).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.6504170890091214, phi=0.5364505757789483).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.568670761625143, phi=0.5437849784272266).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5465254433686004, phi=0.5557344191201743).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.2913067233725621, phi=0.47167387649514164).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5381480272852108, phi=0.6090669301824796).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5830570458423034, phi=0.5523056687078169).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.546350568481482, phi=0.4879631369282715).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.5326394193330342, phi=0.4395048201320801).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5745817206538912, phi=0.5078468553250708).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5287959808590572, phi=0.5477757594186843).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4855370661448744, phi=0.5162805489497694).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.527373036078513, phi=0.5115160136560014).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4411810311735318, phi=0.4660991447993951).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.4000192936015627, phi=0.45553928758281914).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5800842999687046, phi=0.45282693980374283).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.518789513006022, phi=0.48179520284759936).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.529911511830527, phi=0.5594440380846348).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.4246757982684226, phi=0.4944809315999467).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.5296678467857068, phi=0.5740197418485297).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.4984631547270855, phi=0.49206955088399174).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.2089621190327406).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=2.3679632133538817).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-1.836367047964991).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.112920567322682).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.513515124184501).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.458569439617129).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.336122649656808).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.509550700687683).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.7563741935812445).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.5580439702151594).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.2599541836050676).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.272049262238859).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.2579085413115223).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.171271500786176).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.8034922621690797).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=2.937190527560155).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.2515938568548677).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.4328364896860535).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.0640069645082875).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.6401391397375633).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.5600199216855586).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-2.6699368113045594).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.8322428692168204).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.7863909758073149).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.8954718672842006).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.225798767222173).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.5289904259469935).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.761649901992115).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.5876289953142335).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.6688755365766073).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=1.5913632571704357).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-0.8707622124368513).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.686119194414637).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.7581639487130829).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-2.192908102028646).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.203799860595389).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.2740137627486021).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.066149556680692).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.58137837640065).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.9529292695288376).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=3.0652853590094935).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=3.1254309432371983).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
    ]
)

