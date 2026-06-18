from app.models.db import engine, Base, SessionLocal

# Ensure tables exist
Base.metadata.create_all(bind=engine)

# Create DB session
db = SessionLocal()

from app.agent.agent_service import AgentService
from app.repository.booking_repo import BookingRepository

svc = AgentService()
print('Running AgentService.run to book 1 pax 3A Bangalore->Chennai')
reply = svc.run(
    user_message='Book me a 3A ticket from Bangalore to Chennai for 1 passenger',
    conversation_history=[],
    db=db,
    session_id='test_session',
    user_id=1,
)
print('--- AGENT REPLY ---')
print(reply)
print('--- BOOKINGS IN DB ---')
repo = BookingRepository()
bookings = repo.list_by_user(1, db)
print('count=', len(bookings))
for b in bookings:
    try:
        print({
            'id': b.id,
            'user_id': b.user_id,
            'train_number': b.train_number,
            'passenger_count': b.passenger_count,
            'travel_class': b.travel_class,
            'travel_date': str(b.travel_date),
            'status': b.status,
            'created_at': str(b.created_at),
        })
    except Exception as e:
        print('error printing booking', e)

db.close()
